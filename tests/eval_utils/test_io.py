"""Tests for eval_utils.io module."""

import unittest
import tempfile
import os
import csv

from agent_expt_suite.eval_utils.io import save_status_differences_to_csv


class TestIO(unittest.TestCase):
    """Test cases for eval_utils.io module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample status groups
        self.sample_status_groups = {
            ('pass', 'fail'): ['problem_1'],
            ('fail', 'pass'): ['problem_2'],
            ('pass', 'pass'): ['problem_3']
        }

    def test_save_status_differences_to_csv(self):
        """Test saving status differences to CSV."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            csv_file = f.name
        
        try:
            save_status_differences_to_csv(
                self.sample_status_groups,
                csv_file,
                'exp1_status',
                'exp2_status'
            )
            
            # Read the CSV file and verify contents
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader) 
                
                # Should only have differences (not pass, pass)
                self.assertEqual(len(rows), 2)
                
                # Check first row
                self.assertEqual(rows[0]['problem_id'], 'problem_1')
                self.assertEqual(rows[0]['exp1_status'], 'pass')
                self.assertEqual(rows[0]['exp2_status'], 'fail')
                
                # Check second row  
                self.assertEqual(rows[1]['problem_id'], 'problem_2')
                self.assertEqual(rows[1]['exp1_status'], 'fail')
                self.assertEqual(rows[1]['exp2_status'], 'pass')
                
        finally:
            os.unlink(csv_file)


if __name__ == '__main__':
    unittest.main()
