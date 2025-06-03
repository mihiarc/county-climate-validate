#!/usr/bin/env python3
"""
Climate Data QA/QC Validator
Comprehensive validation for temporal and spatial consistency of climate metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('climate_qaqc.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ClimateDataValidator:
    """
    Comprehensive QA/QC validator for climate datasets organized by county and year
    """
    
    def __init__(self, data_file: str):
        """Initialize validator with climate data file"""
        self.data_file = data_file
        self.df = None
        self.issues = {
            'critical': [],
            'warning': [],
            'info': []
        }
        self.validation_results = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and basic preprocessing of climate data"""
        logger.info(f"Loading climate data from {self.data_file}")
        
        try:
            self.df = pd.read_csv(self.data_file)
            logger.info(f"Loaded {len(self.df):,} records with {len(self.df.columns)} columns")
            
            # Basic data info
            logger.info(f"Date range: {self.df['year'].min()} - {self.df['year'].max()}")
            logger.info(f"Unique counties: {self.df['GEOID'].nunique():,}")
            logger.info(f"Scenarios: {', '.join(self.df['scenario'].unique())}")
            
            return self.df
            
        except Exception as e:
            self.issues['critical'].append(f"Failed to load data: {str(e)}")
            raise
    
    def validate_data_completeness(self) -> Dict:
        """Check for missing data patterns"""
        logger.info("Validating data completeness...")
        
        results = {}
        
        # Check for missing values
        missing_summary = self.df.isnull().sum()
        missing_pct = (missing_summary / len(self.df)) * 100
        
        results['missing_values'] = missing_summary[missing_summary > 0].to_dict()
        results['missing_percentages'] = missing_pct[missing_pct > 0].to_dict()
        
        # Check for duplicate records
        duplicates = self.df.duplicated(subset=['GEOID', 'year', 'scenario']).sum()
        results['duplicate_records'] = duplicates
        
        if duplicates > 0:
            self.issues['critical'].append(f"Found {duplicates} duplicate records (same GEOID, year, scenario)")
        
        # Check temporal completeness for each county-scenario combination
        expected_years = set(range(1980, 2101))
        incomplete_series = []
        
        for (geoid, scenario), group in self.df.groupby(['GEOID', 'scenario']):
            years_present = set(group['year'])
            missing_years = expected_years - years_present
            
            if missing_years:
                incomplete_series.append({
                    'GEOID': geoid,
                    'scenario': scenario,
                    'missing_years': len(missing_years),
                    'total_expected': len(expected_years)
                })
        
        results['incomplete_time_series'] = len(incomplete_series)
        results['incomplete_series_details'] = incomplete_series[:100]  # First 100 examples
        
        if incomplete_series:
            self.issues['warning'].append(f"Found {len(incomplete_series)} county-scenario combinations with incomplete time series")
        
        return results
    
    def validate_spatial_consistency(self) -> Dict:
        """Validate spatial patterns and county coverage"""
        logger.info("Validating spatial consistency...")
        
        results = {}
        
        # Check county coverage across scenarios
        county_scenario_coverage = self.df.groupby('GEOID')['scenario'].apply(lambda x: set(x.unique())).to_dict()
        
        all_scenarios = set(self.df['scenario'].unique())
        counties_missing_scenarios = {}
        
        for geoid, scenarios in county_scenario_coverage.items():
            missing = all_scenarios - scenarios
            if missing:
                counties_missing_scenarios[geoid] = list(missing)
        
        results['counties_missing_scenarios'] = counties_missing_scenarios
        
        if counties_missing_scenarios:
            self.issues['warning'].append(f"Found {len(counties_missing_scenarios)} counties missing one or more scenarios")
        
        # Expected CONUS county count (continental US only, excluding territories and non-CONUS)
        expected_conus_counties = 3109
        actual_counties = self.df['GEOID'].nunique()
        results['expected_conus_counties'] = expected_conus_counties
        results['actual_counties'] = actual_counties
        results['missing_counties_count'] = expected_conus_counties - actual_counties
        
        if actual_counties != expected_conus_counties:
            self.issues['info'].append(f"Dataset contains {actual_counties} counties, expected {expected_conus_counties} for CONUS. Missing {expected_conus_counties - actual_counties} counties (likely non-CONUS territories)")
        
        # Check for spatial outliers in climate metrics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        climate_metrics = [col for col in numeric_cols if col not in ['GEOID', 'year', 'pixel_count']]
        
        spatial_outliers = {}
        for metric in climate_metrics:
            # For each year and scenario, identify counties with extreme values
            outlier_counties = []
            
            for (year, scenario), group in self.df.groupby(['year', 'scenario']):
                Q1 = group[metric].quantile(0.25)
                Q3 = group[metric].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                extreme_outliers = group[
                    (group[metric] < lower_bound) | (group[metric] > upper_bound)
                ]['GEOID'].tolist()
                
                if extreme_outliers:
                    outlier_counties.extend(extreme_outliers)
            
            if outlier_counties:
                # Count frequency of outliers per county
                outlier_freq = pd.Series(outlier_counties).value_counts()
                spatial_outliers[metric] = outlier_freq.head(10).to_dict()  # Top 10 outlier counties
        
        results['spatial_outliers'] = spatial_outliers
        
        return results
    
    def validate_temporal_consistency(self) -> Dict:
        """Validate temporal trends and consistency"""
        logger.info("Validating temporal consistency...")
        
        results = {}
        
        # Check for unrealistic year-to-year changes
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        climate_metrics = [col for col in numeric_cols if col not in ['GEOID', 'year', 'pixel_count']]
        
        temporal_issues = {}
        
        for metric in climate_metrics:
            extreme_changes = []
            
            for (geoid, scenario), group in self.df.groupby(['GEOID', 'scenario']):
                if len(group) < 2:
                    continue
                
                group_sorted = group.sort_values('year')
                year_diff = group_sorted[metric].diff()
                
                # Define extreme change thresholds based on metric type
                if 'temp' in metric.lower():
                    threshold = 10  # 10°C change year-to-year is extreme
                elif 'precip' in metric.lower():
                    threshold = group_sorted[metric].std() * 4  # 4 standard deviations
                else:
                    threshold = group_sorted[metric].std() * 3  # 3 standard deviations
                
                extreme_mask = abs(year_diff) > threshold
                if extreme_mask.any():
                    extreme_years = group_sorted[extreme_mask]['year'].tolist()
                    extreme_changes.extend([
                        {'GEOID': geoid, 'scenario': scenario, 'year': year, 'change': change}
                        for year, change in zip(extreme_years, year_diff[extreme_mask])
                    ])
            
            if extreme_changes:
                temporal_issues[metric] = extreme_changes[:50]  # First 50 examples
        
        results['extreme_temporal_changes'] = temporal_issues
        
        # Check for monotonic trends where they shouldn't exist (like temperature)
        trend_issues = {}
        for metric in ['annual_mean_temp_c', 'annual_precipitation_mm']:
            if metric in self.df.columns:
                monotonic_counties = []
                
                for (geoid, scenario), group in self.df.groupby(['GEOID', 'scenario']):
                    if len(group) < 10:  # Need at least 10 years
                        continue
                    
                    group_sorted = group.sort_values('year')
                    values = group_sorted[metric].values
                    
                    # Check if strictly increasing or decreasing
                    if np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0):
                        monotonic_counties.append({'GEOID': geoid, 'scenario': scenario})
                
                if monotonic_counties:
                    trend_issues[metric] = monotonic_counties[:20]  # First 20 examples
        
        results['suspicious_monotonic_trends'] = trend_issues
        
        return results
    
    def validate_logical_relationships(self) -> Dict:
        """Validate logical relationships between climate variables"""
        logger.info("Validating logical relationships between variables...")
        
        results = {}
        logical_errors = []
        
        # Check temperature relationships
        if all(col in self.df.columns for col in ['annual_min_temp_c', 'annual_max_temp_c', 'annual_mean_temp_c']):
            # Mean should be between min and max
            temp_logic_errors = self.df[
                (self.df['annual_mean_temp_c'] < self.df['annual_min_temp_c']) |
                (self.df['annual_mean_temp_c'] > self.df['annual_max_temp_c'])
            ]
            
            if len(temp_logic_errors) > 0:
                logical_errors.append({
                    'variable': 'temperature_relationship',
                    'error_count': len(temp_logic_errors),
                    'description': 'Mean temperature outside min-max range',
                    'examples': temp_logic_errors[['GEOID', 'year', 'scenario', 'annual_min_temp_c', 
                                                 'annual_mean_temp_c', 'annual_max_temp_c']].head(10).to_dict('records')
                })
        
        # Check precipitation days vs total precipitation
        if all(col in self.df.columns for col in ['annual_precipitation_mm', 'high_precip_days_95th']):
            # High precipitation days should be reasonable relative to total precipitation
            precip_outliers = self.df[
                (self.df['high_precip_days_95th'] > 100) |  # More than 100 high precip days
                ((self.df['annual_precipitation_mm'] < 100) & (self.df['high_precip_days_95th'] > 20))  # Low precip but many high precip days
            ]
            
            if len(precip_outliers) > 0:
                logical_errors.append({
                    'variable': 'precipitation_relationship',
                    'error_count': len(precip_outliers),
                    'description': 'Inconsistent precipitation days vs total precipitation',
                    'examples': precip_outliers[['GEOID', 'year', 'scenario', 'annual_precipitation_mm', 
                                               'high_precip_days_95th']].head(10).to_dict('records')
                })
        
        # Check cold/hot days relationships
        if all(col in self.df.columns for col in ['cold_days_0c', 'hot_days_30c', 'annual_mean_temp_c']):
            # Hot days should be rare in very cold climates
            hot_cold_errors = self.df[
                (self.df['annual_mean_temp_c'] < 0) & (self.df['hot_days_30c'] > 30)
            ]
            
            if len(hot_cold_errors) > 0:
                logical_errors.append({
                    'variable': 'temperature_days_relationship',
                    'error_count': len(hot_cold_errors),
                    'description': 'Many hot days in cold climate regions',
                    'examples': hot_cold_errors[['GEOID', 'year', 'scenario', 'annual_mean_temp_c', 
                                               'hot_days_30c']].head(10).to_dict('records')
                })
        
        results['logical_errors'] = logical_errors
        
        return results
    
    def validate_physical_plausibility(self) -> Dict:
        """Check for physically implausible values"""
        logger.info("Validating physical plausibility...")
        
        results = {}
        plausibility_errors = []
        
        # Define plausible ranges for climate variables
        plausible_ranges = {
            'annual_mean_temp_c': (-50, 50),
            'annual_min_temp_c': (-80, 40),
            'annual_max_temp_c': (-30, 60),
            'annual_precipitation_mm': (0, 10000),
            'hot_days_30c': (0, 365),
            'cold_days_0c': (0, 365),
            'growing_degree_days': (0, 8000)
        }
        
        for variable, (min_val, max_val) in plausible_ranges.items():
            if variable in self.df.columns:
                out_of_range = self.df[
                    (self.df[variable] < min_val) | (self.df[variable] > max_val)
                ]
                
                if len(out_of_range) > 0:
                    plausibility_errors.append({
                        'variable': variable,
                        'error_count': len(out_of_range),
                        'expected_range': f"{min_val} to {max_val}",
                        'actual_range': f"{out_of_range[variable].min():.2f} to {out_of_range[variable].max():.2f}",
                        'examples': out_of_range[['GEOID', 'year', 'scenario', variable]].head(10).to_dict('records')
                    })
        
        results['plausibility_errors'] = plausibility_errors
        
        return results
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics for the dataset"""
        logger.info("Generating summary statistics...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        climate_metrics = [col for col in numeric_cols if col not in ['GEOID', 'year', 'pixel_count']]
        
        summary_stats = {}
        
        for metric in climate_metrics:
            stats = {
                'count': self.df[metric].count(),
                'mean': self.df[metric].mean(),
                'std': self.df[metric].std(),
                'min': self.df[metric].min(),
                'q25': self.df[metric].quantile(0.25),
                'median': self.df[metric].median(),
                'q75': self.df[metric].quantile(0.75),
                'max': self.df[metric].max()
            }
            summary_stats[metric] = {k: round(v, 3) if not pd.isna(v) else None for k, v in stats.items()}
        
        return summary_stats
    
    def run_full_validation(self) -> Dict:
        """Run complete QA/QC validation suite"""
        logger.info("Starting comprehensive QA/QC validation...")
        
        # Load data
        self.load_data()
        
        # Run all validation checks
        self.validation_results = {
            'data_completeness': self.validate_data_completeness(),
            'spatial_consistency': self.validate_spatial_consistency(),
            'temporal_consistency': self.validate_temporal_consistency(),
            'logical_relationships': self.validate_logical_relationships(),
            'physical_plausibility': self.validate_physical_plausibility(),
            'summary_statistics': self.generate_summary_statistics()
        }
        
        # Generate overall assessment
        self._generate_overall_assessment()
        
        return self.validation_results
    
    def _generate_overall_assessment(self):
        """Generate overall data quality assessment"""
        total_critical = len(self.issues['critical'])
        total_warnings = len(self.issues['warning'])
        total_info = len(self.issues['info'])
        
        if total_critical > 0:
            quality_score = "POOR"
            recommendation = "Critical issues found. Data should not be used for analysis until resolved."
        elif total_warnings > 10:
            quality_score = "FAIR"
            recommendation = "Multiple warnings found. Review and address before using for analysis."
        elif total_warnings > 0:
            quality_score = "GOOD"
            recommendation = "Minor issues found. Generally suitable for analysis with caution."
        else:
            quality_score = "EXCELLENT"
            recommendation = "No significant issues found. Data appears suitable for analysis."
        
        self.validation_results['overall_assessment'] = {
            'quality_score': quality_score,
            'recommendation': recommendation,
            'critical_issues': total_critical,
            'warnings': total_warnings,
            'info_items': total_info,
            'detailed_issues': self.issues
        }
    
    def save_validation_report(self, output_file: str = None):
        """Save detailed validation report"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"climate_qaqc_report_{timestamp}.txt"
        
        with open(output_file, 'w') as f:
            f.write("CLIMATE DATA QA/QC VALIDATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.data_file}\n\n")
            
            # Overall assessment
            if 'overall_assessment' in self.validation_results:
                assessment = self.validation_results['overall_assessment']
                f.write("OVERALL ASSESSMENT\n")
                f.write("-" * 20 + "\n")
                f.write(f"Quality Score: {assessment['quality_score']}\n")
                f.write(f"Recommendation: {assessment['recommendation']}\n")
                f.write(f"Critical Issues: {assessment['critical_issues']}\n")
                f.write(f"Warnings: {assessment['warnings']}\n\n")
            
            # Detailed results
            for section, results in self.validation_results.items():
                if section != 'overall_assessment':
                    f.write(f"{section.upper().replace('_', ' ')}\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"{results}\n\n")
        
        logger.info(f"Validation report saved to {output_file}")
    
    def create_visualization_plots(self, output_dir: str = "qaqc_plots"):
        """Create visualization plots for QA/QC results"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Plot 1: Data completeness by scenario and year
        plt.figure(figsize=(12, 6))
        completeness = self.df.groupby(['year', 'scenario']).size().unstack(fill_value=0)
        completeness.plot(kind='line', marker='o')
        plt.title('Data Completeness by Year and Scenario')
        plt.xlabel('Year')
        plt.ylabel('Number of Counties')
        plt.legend(title='Scenario')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/data_completeness.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Temperature trends by scenario
        if 'annual_mean_temp_c' in self.df.columns:
            plt.figure(figsize=(12, 6))
            temp_trends = self.df.groupby(['year', 'scenario'])['annual_mean_temp_c'].mean().unstack()
            temp_trends.plot(kind='line', marker='o')
            plt.title('Average Annual Temperature Trends by Scenario')
            plt.xlabel('Year')
            plt.ylabel('Temperature (°C)')
            plt.legend(title='Scenario')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/temperature_trends.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualization plots saved to {output_dir}/")


def main():
    """Main function to run climate data validation"""
    # Initialize validator
    validator = ClimateDataValidator("county_climate_metrics_complete_1980_2100.csv")
    
    # Run full validation
    results = validator.run_full_validation()
    
    # Save report
    validator.save_validation_report()
    
    # Create visualizations
    validator.create_visualization_plots()
    
    # Print summary
    print("\nVALIDATION SUMMARY")
    print("=" * 40)
    assessment = results.get('overall_assessment', {})
    print(f"Quality Score: {assessment.get('quality_score', 'Unknown')}")
    print(f"Critical Issues: {assessment.get('critical_issues', 0)}")
    print(f"Warnings: {assessment.get('warnings', 0)}")
    print(f"\nRecommendation: {assessment.get('recommendation', 'No assessment available')}")


if __name__ == "__main__":
    main() 