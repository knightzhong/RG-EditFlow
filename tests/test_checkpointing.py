import unittest

from src.checkpointing import should_save_epoch_checkpoint


class CheckpointScheduleTests(unittest.TestCase):
    def test_save_every_zero_disables_epoch_checkpoints(self):
        self.assertFalse(should_save_epoch_checkpoint(epoch=10, save_every=0))
        self.assertFalse(should_save_epoch_checkpoint(epoch=100, save_every=0))

    def test_positive_save_every_keeps_periodic_checkpoints(self):
        self.assertFalse(should_save_epoch_checkpoint(epoch=9, save_every=10))
        self.assertTrue(should_save_epoch_checkpoint(epoch=10, save_every=10))
        self.assertTrue(should_save_epoch_checkpoint(epoch=20, save_every=10))


if __name__ == "__main__":
    unittest.main()
