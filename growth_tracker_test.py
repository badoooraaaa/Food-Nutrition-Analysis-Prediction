import unittest
from who_data import calculate_percentile

class TestGrowthTracker(unittest.TestCase):
    
    def test_percentile_median_boy(self):
        # Median weight for 6 month old boy is 7.9kg
        perc, z_score, status = calculate_percentile(7.9, 6.0, 'boy', 'weight')
        self.assertAlmostEqual(perc, 50.0, places=1)
        self.assertAlmostEqual(z_score, 0.0, places=1)
        self.assertEqual(status, 'Normal')
        
    def test_percentile_below_average_girl(self):
        # 12 month old girl, P15 height is 71.4cm. Value 70cm should be < P15
        perc, z_score, status = calculate_percentile(70.0, 12.0, 'girl', 'height')
        self.assertTrue(perc < 15.0)
        self.assertTrue(z_score < -1.0)
        self.assertIn('Stunted', status) # < 15 is Stunted or Severely Stunted
        
    def test_percentile_above_average_boy(self):
        # 24 month old boy, P85 weight is 13.6kg. Value 14kg should be > P85
        perc, z_score, status = calculate_percentile(14.0, 24.0, 'boy', 'weight')
        self.assertTrue(perc > 85.0)
        self.assertTrue(z_score > 1.0)
        self.assertIn('Overweight', status)

if __name__ == '__main__':
    unittest.main()
