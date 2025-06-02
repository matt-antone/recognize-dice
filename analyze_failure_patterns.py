#!/usr/bin/env python3
"""
Analyze failure patterns from Kaggle dataset testing results.
Identify specific causes and actionable insights for improvement.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

class FailurePatternAnalyzer:
    """Comprehensive analysis of dice detection failure patterns."""
    
    def __init__(self, results_file):
        self.results_file = results_file
        self.results_data = None
        self.failures = []
        self.successes = []
        
    def load_results(self):
        """Load the JSON results file."""
        print(f"Loading results from: {self.results_file}")
        
        with open(self.results_file, 'r') as f:
            self.results_data = json.load(f)
        
        print(f"✅ Loaded {self.results_data['total_images_tested']} test results")
        
        # Separate failures and successes
        for result in self.results_data['results']:
            if result['detection_count'] == 0:
                self.failures.append(result)
            else:
                self.successes.append(result)
        
        print(f"Failures: {len(self.failures)}")
        print(f"Successes: {len(self.successes)}")
    
    def analyze_image_characteristics(self):
        """Analyze image size patterns between failures and successes."""
        print("\n=== IMAGE CHARACTERISTICS ANALYSIS ===")
        
        failure_sizes = [tuple(result['image_size']) for result in self.failures]
        success_sizes = [tuple(result['image_size']) for result in self.successes]
        
        failure_size_counts = Counter(failure_sizes)
        success_size_counts = Counter(success_sizes)
        
        print("Top failure image sizes:")
        for size, count in failure_size_counts.most_common(5):
            print(f"  {size[1]}x{size[0]}: {count} failures")
        
        print("Top success image sizes:")
        for size, count in success_size_counts.most_common(5):
            print(f"  {size[1]}x{size[0]}: {count} successes")
        
        # Calculate success rates by image size
        all_sizes = set(failure_sizes + success_sizes)
        print("\nSuccess rate by image size:")
        for size in sorted(all_sizes):
            failures = failure_size_counts.get(size, 0)
            successes = success_size_counts.get(size, 0)
            total = failures + successes
            success_rate = (successes / total) * 100 if total > 0 else 0
            if total >= 5:  # Only show sizes with meaningful sample sizes
                print(f"  {size[1]}x{size[0]}: {success_rate:.1f}% ({successes}/{total})")
    
    def analyze_filename_patterns(self):
        """Look for patterns in filenames that correlate with failures."""
        print("\n=== FILENAME PATTERN ANALYSIS ===")
        
        # Extract timestamp and other patterns from filenames
        failure_times = []
        success_times = []
        
        for result in self.failures:
            filename = result['image_name']
            # Extract time pattern IMG_YYYYMMDD_HHMMSS.jpg
            if 'IMG_' in filename:
                try:
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        time_part = parts[2].replace('.jpg', '')
                        hour = int(time_part[:2]) if len(time_part) >= 2 else 0
                        failure_times.append(hour)
                except:
                    pass
        
        for result in self.successes:
            filename = result['image_name']
            if 'IMG_' in filename:
                try:
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        time_part = parts[2].replace('.jpg', '')
                        hour = int(time_part[:2]) if len(time_part) >= 2 else 0
                        success_times.append(hour)
                except:
                    pass
        
        if failure_times and success_times:
            failure_hour_counts = Counter(failure_times)
            success_hour_counts = Counter(success_times)
            
            print("Time-based patterns (by hour):")
            all_hours = set(failure_times + success_times)
            for hour in sorted(all_hours):
                failures = failure_hour_counts.get(hour, 0)
                successes = success_hour_counts.get(hour, 0)
                total = failures + successes
                success_rate = (successes / total) * 100 if total > 0 else 0
                if total >= 3:
                    print(f"  {hour:02d}:xx - {success_rate:.1f}% success ({successes}/{total})")
        
        # Check for date patterns
        failure_dates = []
        success_dates = []
        
        for result in self.failures:
            filename = result['image_name']
            if 'IMG_' in filename:
                try:
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        date_part = parts[1]
                        failure_dates.append(date_part)
                except:
                    pass
        
        for result in self.successes:
            filename = result['image_name']
            if 'IMG_' in filename:
                try:
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        date_part = parts[1]
                        success_dates.append(date_part)
                except:
                    pass
        
        if failure_dates and success_dates:
            failure_date_counts = Counter(failure_dates)
            success_date_counts = Counter(success_dates)
            
            print("\nDate-based patterns:")
            all_dates = set(failure_dates + success_dates)
            for date in sorted(all_dates):
                failures = failure_date_counts.get(date, 0)
                successes = success_date_counts.get(date, 0)
                total = failures + successes
                success_rate = (successes / total) * 100 if total > 0 else 0
                print(f"  {date}: {success_rate:.1f}% success ({successes}/{total})")
    
    def analyze_processing_time_patterns(self):
        """Look for processing time differences between failures and successes."""
        print("\n=== PROCESSING TIME ANALYSIS ===")
        
        failure_times = [result['processing_time_ms'] for result in self.failures]
        success_times = [result['processing_time_ms'] for result in self.successes]
        
        if failure_times and success_times:
            print(f"Failure processing times:")
            print(f"  Average: {np.mean(failure_times):.1f}ms")
            print(f"  Median: {np.median(failure_times):.1f}ms")
            print(f"  Range: {min(failure_times):.1f} - {max(failure_times):.1f}ms")
            
            print(f"Success processing times:")
            print(f"  Average: {np.mean(success_times):.1f}ms")
            print(f"  Median: {np.median(success_times):.1f}ms")
            print(f"  Range: {min(success_times):.1f} - {max(success_times):.1f}ms")
            
            # Statistical test
            time_diff = np.mean(success_times) - np.mean(failure_times)
            print(f"\nTime difference: Successes take {time_diff:+.1f}ms more on average")
            
            if abs(time_diff) > 5:
                if time_diff > 0:
                    print("💡 INSIGHT: Successful detections take longer (more processing)")
                else:
                    print("💡 INSIGHT: Failed detections take longer (struggling to find dice)")
    
    def analyze_detection_value_patterns(self):
        """Analyze which dice values are most/least detected."""
        print("\n=== DETECTION VALUE ANALYSIS ===")
        
        value_counts = defaultdict(int)
        total_detections = 0
        
        for result in self.successes:
            for detection in result['detections']:
                value = detection['value']
                value_counts[value] += 1
                total_detections += 1
        
        print("Detected value distribution:")
        for value in sorted(value_counts.keys()):
            count = value_counts[value]
            percentage = (count / total_detections) * 100
            print(f"  Value {value}: {count} detections ({percentage:.1f}%)")
        
        # Identify problematic values
        expected_percentage = 100 / 6  # ~16.67% if uniform
        print(f"\nExpected uniform distribution: {expected_percentage:.1f}% per value")
        
        print("Deviations from expected:")
        for value in range(1, 7):
            count = value_counts.get(value, 0)
            actual_percentage = (count / total_detections) * 100 if total_detections > 0 else 0
            deviation = actual_percentage - expected_percentage
            
            if abs(deviation) > 5:  # Significant deviation
                if deviation > 0:
                    print(f"  Value {value}: OVER-detected by {deviation:+.1f}%")
                else:
                    print(f"  Value {value}: UNDER-detected by {deviation:+.1f}%")
    
    def analyze_method_patterns(self):
        """Analyze which detection methods work best."""
        print("\n=== DETECTION METHOD ANALYSIS ===")
        
        method_counts = defaultdict(int)
        method_confidence = defaultdict(list)
        method_areas = defaultdict(list)
        
        for result in self.successes:
            for detection in result['detections']:
                method = detection['method']
                confidence = detection['confidence']
                area = detection['area']
                
                method_counts[method] += 1
                method_confidence[method].append(confidence)
                method_areas[method].append(area)
        
        print("Detection method performance:")
        for method in method_counts:
            count = method_counts[method]
            avg_confidence = np.mean(method_confidence[method])
            avg_area = np.mean(method_areas[method])
            
            print(f"  {method}:")
            print(f"    Count: {count}")
            print(f"    Avg confidence: {avg_confidence:.3f}")
            print(f"    Avg area: {avg_area:.0f} pixels")
    
    def find_failure_clusters(self):
        """Group failures by similar characteristics to find patterns."""
        print("\n=== FAILURE CLUSTERING ANALYSIS ===")
        
        # Group by processing time ranges
        time_ranges = [
            (0, 45, "Fast (<45ms)"),
            (45, 50, "Medium (45-50ms)"),
            (50, 60, "Slow (50-60ms)"),
            (60, 200, "Very Slow (>60ms)")
        ]
        
        print("Failures by processing time:")
        for min_time, max_time, label in time_ranges:
            count = sum(1 for result in self.failures 
                       if min_time <= result['processing_time_ms'] < max_time)
            if count > 0:
                print(f"  {label}: {count} failures")
        
        # Group by image size
        print("\nFailures by image size category:")
        size_categories = defaultdict(list)
        
        for result in self.failures:
            h, w = result['image_size']
            total_pixels = h * w
            
            if total_pixels < 1000000:  # < 1MP
                category = "Small"
            elif total_pixels < 4000000:  # < 4MP
                category = "Medium"
            else:
                category = "Large"
            
            size_categories[category].append(result)
        
        for category, results in size_categories.items():
            print(f"  {category} images: {len(results)} failures")
    
    def generate_actionable_insights(self):
        """Generate specific, actionable recommendations based on analysis."""
        print("\n" + "=" * 60)
        print("ACTIONABLE INSIGHTS & RECOMMENDATIONS")
        print("=" * 60)
        
        insights = []
        
        # Value 6 detection issue
        value_counts = defaultdict(int)
        total_detections = 0
        for result in self.successes:
            for detection in result['detections']:
                value_counts[detection['value']] += 1
                total_detections += 1
        
        if value_counts.get(6, 0) < total_detections * 0.05:  # Less than 5%
            insights.append({
                'priority': 'CRITICAL',
                'issue': 'Value 6 Under-Detection',
                'evidence': f'Only {value_counts.get(6, 0)} of {total_detections} detections ({(value_counts.get(6, 0)/total_detections)*100:.1f}%)',
                'action': 'Investigate pip pattern recognition for value 6 specifically'
            })
        
        # High failure rate
        failure_rate = len(self.failures) / (len(self.failures) + len(self.successes)) * 100
        if failure_rate > 50:
            insights.append({
                'priority': 'HIGH',
                'issue': 'High Overall Failure Rate',
                'evidence': f'{failure_rate:.1f}% of images failed detection',
                'action': 'Implement angle-agnostic detection improvements'
            })
        
        # Processing time differences
        if self.failures and self.successes:
            failure_times = [r['processing_time_ms'] for r in self.failures]
            success_times = [r['processing_time_ms'] for r in self.successes]
            time_diff = np.mean(success_times) - np.mean(failure_times)
            
            if time_diff > 10:
                insights.append({
                    'priority': 'MEDIUM',
                    'issue': 'Successful Detections Take Longer',
                    'evidence': f'Successes average {time_diff:.1f}ms longer processing',
                    'action': 'Complex detections work better - consider relaxing early-exit conditions'
                })
        
        # Value distribution skew
        value_1_pct = (value_counts.get(1, 0) / total_detections) * 100 if total_detections > 0 else 0
        if value_1_pct > 35:  # Over-detection of 1s
            insights.append({
                'priority': 'MEDIUM',
                'issue': 'Value 1 Over-Detection',
                'evidence': f'Value 1 represents {value_1_pct:.1f}% of detections (expected ~16.7%)',
                'action': 'Review pip counting algorithm - may be detecting single dots incorrectly'
            })
        
        # Print insights
        for insight in sorted(insights, key=lambda x: {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2}.get(x['priority'], 3)):
            print(f"\n🚨 {insight['priority']} PRIORITY")
            print(f"Issue: {insight['issue']}")
            print(f"Evidence: {insight['evidence']}")
            print(f"Action: {insight['action']}")
        
        return insights
    
    def save_analysis_report(self, insights):
        """Save comprehensive analysis report."""
        timestamp = Path(self.results_file).stem.split('_')[-2:]
        report_file = f"failure_analysis_{'_'.join(timestamp)}.txt"
        
        with open(report_file, 'w') as f:
            f.write("DICE DETECTION FAILURE PATTERN ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset: {self.results_data['total_images_tested']} images\n")
            f.write(f"Failures: {len(self.failures)} ({len(self.failures)/(len(self.failures)+len(self.successes))*100:.1f}%)\n")
            f.write(f"Successes: {len(self.successes)} ({len(self.successes)/(len(self.failures)+len(self.successes))*100:.1f}%)\n\n")
            
            f.write("CRITICAL INSIGHTS:\n")
            for insight in insights:
                f.write(f"- {insight['issue']}: {insight['evidence']}\n")
                f.write(f"  Action: {insight['action']}\n\n")
        
        print(f"\n📄 Analysis report saved to: {report_file}")
    
    def run_complete_analysis(self):
        """Run all analysis functions."""
        self.load_results()
        self.analyze_image_characteristics()
        self.analyze_filename_patterns()
        self.analyze_processing_time_patterns()
        self.analyze_detection_value_patterns()
        self.analyze_method_patterns()
        self.find_failure_clusters()
        insights = self.generate_actionable_insights()
        self.save_analysis_report(insights)


def main():
    """Main analysis function."""
    print("🔍 FAILURE PATTERN ANALYSIS")
    print("Analyzing Kaggle dataset test results...")
    print("=" * 50)
    
    # Find the most recent results file
    results_files = list(Path('.').glob('kaggle_dataset_results_*.json'))
    if not results_files:
        print("❌ No results files found")
        print("Run: python3 test_kaggle_dataset.py first")
        return
    
    # Use the most recent file
    results_file = max(results_files, key=lambda p: p.stat().st_mtime)
    
    analyzer = FailurePatternAnalyzer(results_file)
    analyzer.run_complete_analysis()
    
    print("\n✅ Analysis complete!")
    print("Check the generated report file for detailed findings.")


if __name__ == "__main__":
    main() 