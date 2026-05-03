from time import time
from typing import List, Optional

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from torch import distributed as dist

from nnunetv2.training.nnUNetTrainer.fixed_val_tiles import FixedValTileManager
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.git_logging import log_git_context

class nnUNetTrainerFixedValTiles(nnUNetTrainer):
    _old_val_logger_keys = (
        'old_val_mean_fg_dice',
        'old_val_ema_fg_dice',
        'old_val_dice_per_class_or_region',
        'old_val_losses',
    )

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.fixed_val_tile_step_size = 0.5
        # Fixed seed makes the selected validation tile pool reproducible across epochs/runs.
        self.fixed_val_tile_seed = 42
        self.fixed_val_num_tiles = None
        self.fixed_val_manager: Optional[FixedValTileManager] = None
        self._best_old = None
        self._register_fixed_val_logger_keys()
        log_git_context(self)

    def _register_fixed_val_logger_keys(self):
        logging_dict = self.logger.local_logger.my_fantastic_logging
        # These old_val_* series compare against the original random validation sampler without
        # overwriting the fixed-tile metrics stored in the base logger keys.
        for key in self._old_val_logger_keys:
            logging_dict.setdefault(key, [])

    def load_checkpoint(self, filename_or_checkpoint):
        super().load_checkpoint(filename_or_checkpoint)
        old_val_ema_fg_dice = self.logger.get_value('old_val_ema_fg_dice', step=None)
        finite_scores = [float(i) for i in old_val_ema_fg_dice if np.isfinite(i)]
        self._best_old = max(finite_scores) if len(finite_scores) > 0 else None

    def on_train_start(self):
        super().on_train_start()
        self.fixed_val_manager = self._build_fixed_validation_tile_manager()

    def _build_fixed_validation_tile_manager(self) -> FixedValTileManager:
        # It is safe to call do_split again, since it uses fixed seeding
        _, val_keys = self.do_split()
        dataset_val = self.dataset_class(
            self.preprocessed_dataset_folder,
            val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
        )

        patch_size = tuple(int(i) for i in self.configuration_manager.patch_size)
        global_batch_size = self.configuration_manager.batch_size if self.is_ddp else self.batch_size
        num_tiles = self.fixed_val_num_tiles
        if num_tiles is None:
            num_tiles = self.num_val_iterations_per_epoch * global_batch_size
        rank = dist.get_rank() if self.is_ddp else 0
        world_size = dist.get_world_size() if self.is_ddp else 1
        transforms = self.get_validation_transforms(
            self._get_deep_supervision_scales(),
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        manager = FixedValTileManager(
            dataset=dataset_val,
            patch_size=patch_size,
            batch_size=self.batch_size,
            transforms=transforms,
            tile_step_size=self.fixed_val_tile_step_size,
            seed=self.fixed_val_tile_seed,
            num_tiles=num_tiles,
            is_ddp=self.is_ddp,
            rank=rank,
            world_size=world_size,
        )

        stats = manager.stats
        self.print_to_log_file(
            f"Fixed validation tile pool: {stats.num_cases} cases, {stats.num_candidate_tiles} candidate tiles, "
            f"{stats.num_requested_tiles} requested tiles, {stats.num_selected_tiles} selected tiles, "
            f"{stats.num_tiles_on_rank} tiles on rank {self.local_rank}",
        )
        return manager

    def on_old_validation_epoch_end(self, val_outputs: List[dict]):
        # Same aggregation as the base validation epoch end, but written to old_val_* keys
        # so the fixed-tile validation remains the primary logged validation result.
        loss_here, global_dc_per_class, mean_fg_dice = self._compute_validation_metrics(val_outputs)
        self.logger.log('old_val_mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('old_val_dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('old_val_losses', loss_here, self.current_epoch)

        # Mirror MetaLogger's EMA formula for mean_fg_dice, but keep it separate for the
        # old random-validation sampler used by checkpoint_best_old.pth.
        previous_ema_values = self.logger.get_value('old_val_ema_fg_dice', step=None)[:self.current_epoch]
        finite_previous_ema_values = [float(i) for i in previous_ema_values if np.isfinite(i)]
        old_val_ema_fg_dice = finite_previous_ema_values[-1] * 0.9 + 0.1 * mean_fg_dice \
            if len(finite_previous_ema_values) > 0 else mean_fg_dice
        self.logger.log('old_val_ema_fg_dice', old_val_ema_fg_dice, self.current_epoch)

    def _compute_validation_metrics(self, val_outputs: List[dict]):
        # Extracted from nnUNetTrainer.on_validation_epoch_end so we can reuse the exact
        # Dice/loss reduction, including DDP gathers, without logging to the base keys.
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        return loss_here, global_dc_per_class, mean_fg_dice

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.get_value('train_losses', step=-1), decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.get_value('val_losses', step=-1), decimals=4))
        self.print_to_log_file('old_val_loss', np.round(self.logger.get_value('old_val_losses', step=-1), decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.get_value('dice_per_class_or_region', step=-1)])
        self.print_to_log_file('Old val pseudo dice', [np.round(i, decimals=4) for i in
                                                       self.logger.get_value('old_val_dice_per_class_or_region',
                                                                             step=-1)])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.get_value('epoch_end_timestamps', step=-1) - self.logger.get_value('epoch_start_timestamps', step=-1), decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # With fixed validation tiles, raw fixed-val Dice is comparable across epochs; EMA would only add lag.
        fixed_val_score = self.logger.get_value('mean_fg_dice', step=-1)
        # We reuse _best_ema for compatibility with the rest of the training framework
        if self._best_ema is None or fixed_val_score > self._best_ema:
            self._best_ema = fixed_val_score
            self.print_to_log_file(f"New best pseudo Dice: {float(self._best_ema):.4f}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        old_val_ema_fg_dice = self.logger.get_value('old_val_ema_fg_dice', step=-1)
        if self._best_old is None or old_val_ema_fg_dice > self._best_old:
            self._best_old = old_val_ema_fg_dice
            self.print_to_log_file(f"New best old EMA pseudo Dice: {float(self._best_old):.4f}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best_old.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch in self.fixed_val_manager.iter_batches():
                    val_outputs.append(self.validation_step(batch))
                self.on_validation_epoch_end(val_outputs)

                # Run the original random validation sampler as a side-by-side baseline for
                # checkpoint_best_old.pth and comparison against the fixed-tile strategy.
                old_val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    old_val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_old_validation_epoch_end(old_val_outputs)

            self.on_epoch_end()

        self.on_train_end()
