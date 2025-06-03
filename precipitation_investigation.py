#!/usr/bin/env python3
"""
Precipitation Relationships Investigation
Focused analysis of precipitation inconsistencies found in climate QA/QC validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('precipitation_investigation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class PrecipitationInvestigator:
    """
    Focused investigation of precipitation relationship inconsistencies
    """
    
    def __init__(self, data_file: str, output_dir: str = "precipitation_investigation"):
        """Initialize investigator with climate data file"""
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.df = None
        self.problem_records = None
        
    def load_data(self) -> pd.DataFrame:
        """Load climate data"""
        logger.info(f"Loading climate data from {self.data_file}")
        
        try:
            self.df = pd.read_csv(self.data_file)
            logger.info(f"Loaded {len(self.df):,} records with {len(self.df.columns)} columns")
            return self.df
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def identify_precipitation_issues(self):
        """Identify records with precipitation relationship problems"""
        logger.info("Identifying precipitation relationship issues...")
        
        # Check if required columns exist
        required_cols = ['annual_precipitation_mm', 'high_precip_days_95th']
        if not all(col in self.df.columns for col in required_cols):
            logger.error(f"Required columns missing: {required_cols}")
            return
        
        # Define problematic conditions
        conditions = [
            # More than 100 high precip days (impossible)
            self.df['high_precip_days_95th'] > 100,
            
            # Low precipitation but many high precip days
            (self.df['annual_precipitation_mm'] < 100) & (self.df['high_precip_days_95th'] > 20),
            
            # Very high precipitation days relative to total precipitation
            # Using a ratio approach: if you have more than 1 high precip day per 10mm of annual precip
            self.df['high_precip_days_95th'] > (self.df['annual_precipitation_mm'] / 10),
            
            # Extremely high values that seem unrealistic
            self.df['high_precip_days_95th'] > 365,  # More days than in a year
        ]
        
        # Combine all conditions
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = combined_condition | condition
        
        self.problem_records = self.df[combined_condition].copy()
        
        logger.info(f"Found {len(self.problem_records):,} problematic precipitation records")
        
        # Add calculated ratios for analysis
        self.problem_records['precip_per_high_day'] = (
            self.problem_records['annual_precipitation_mm'] / 
            self.problem_records['high_precip_days_95th'].replace(0, np.nan)
        )
        
        self.problem_records['high_day_ratio'] = (
            self.problem_records['high_precip_days_95th'] / 365 * 100
        )
        
        return self.problem_records
    
    def analyze_problem_patterns(self):
        """Analyze patterns in problematic records"""
        logger.info("Analyzing problem patterns...")
        
        if self.problem_records is None or len(self.problem_records) == 0:
            logger.warning("No problem records to analyze")
            return
        
        analysis = {}
        
        # Geographic distribution
        problem_counties = self.problem_records['GEOID'].value_counts()
        analysis['top_problem_counties'] = problem_counties.head(20).to_dict()
        analysis['total_problem_counties'] = len(problem_counties)
        
        # Temporal distribution
        problem_years = self.problem_records['year'].value_counts().sort_index()
        analysis['problem_by_year'] = problem_years.to_dict()
        
        # Scenario distribution
        problem_scenarios = self.problem_records['scenario'].value_counts()
        analysis['problem_by_scenario'] = problem_scenarios.to_dict()
        
        # Severity analysis
        extreme_cases = self.problem_records[self.problem_records['high_precip_days_95th'] > 200]
        analysis['extreme_cases_count'] = len(extreme_cases)
        
        # Statistical summary
        analysis['precipitation_stats'] = {
            'min_annual_precip': self.problem_records['annual_precipitation_mm'].min(),
            'max_annual_precip': self.problem_records['annual_precipitation_mm'].max(),
            'mean_annual_precip': self.problem_records['annual_precipitation_mm'].mean(),
            'min_high_precip_days': self.problem_records['high_precip_days_95th'].min(),
            'max_high_precip_days': self.problem_records['high_precip_days_95th'].max(),
            'mean_high_precip_days': self.problem_records['high_precip_days_95th'].mean()
        }
        
        return analysis
    
    def create_diagnostic_plots(self):
        """Create diagnostic plots for precipitation issues"""
        logger.info("Creating diagnostic plots...")
        
        if self.problem_records is None or len(self.problem_records) == 0:
            logger.warning("No problem records to plot")
            return
        
        # Create comprehensive diagnostic dashboard
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Precipitation Relationships Investigation', fontsize=16, fontweight='bold')
        
        # Plot 1: Scatter plot of annual precip vs high precip days (all data)
        sample_normal = self.df.sample(n=min(10000, len(self.df)))
        axes[0,0].scatter(sample_normal['annual_precipitation_mm'], 
                         sample_normal['high_precip_days_95th'], 
                         alpha=0.3, s=1, color='lightblue', label='Normal data')
        axes[0,0].scatter(self.problem_records['annual_precipitation_mm'], 
                         self.problem_records['high_precip_days_95th'], 
                         alpha=0.7, s=3, color='red', label='Problem records')
        axes[0,0].set_xlabel('Annual Precipitation (mm)')
        axes[0,0].set_ylabel('High Precipitation Days (95th percentile)')
        axes[0,0].set_title('Annual Precipitation vs High Precipitation Days')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Problem records by county
        problem_counties = self.problem_records['GEOID'].value_counts().head(20)
        problem_counties.plot(kind='bar', ax=axes[0,1], color='salmon')
        axes[0,1].set_title('Top 20 Counties with Precipitation Issues')
        axes[0,1].set_xlabel('County GEOID')
        axes[0,1].set_ylabel('Number of Problem Records')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Problem records by year
        problem_years = self.problem_records['year'].value_counts().sort_index()
        axes[0,2].plot(problem_years.index, problem_years.values, marker='o', linewidth=2)
        axes[0,2].set_title('Problem Records by Year')
        axes[0,2].set_xlabel('Year')
        axes[0,2].set_ylabel('Number of Problem Records')
        axes[0,2].grid(True, alpha=0.3)
        
        # Plot 4: Problem records by scenario
        problem_scenarios = self.problem_records['scenario'].value_counts()
        axes[1,0].pie(problem_scenarios.values, labels=problem_scenarios.index, autopct='%1.1f%%')
        axes[1,0].set_title('Problem Records by Scenario')
        
        # Plot 5: Distribution of high precip days in problem records
        axes[1,1].hist(self.problem_records['high_precip_days_95th'], bins=50, 
                      color='orange', alpha=0.7, edgecolor='black')
        axes[1,1].set_xlabel('High Precipitation Days')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Distribution of High Precip Days (Problem Records)')
        axes[1,1].grid(True, alpha=0.3)
        
        # Plot 6: Precipitation per high precip day ratio
        valid_ratios = self.problem_records['precip_per_high_day'].dropna()
        if len(valid_ratios) > 0:
            axes[1,2].hist(valid_ratios, bins=50, color='purple', alpha=0.7, edgecolor='black')
            axes[1,2].set_xlabel('Precipitation per High Precip Day (mm/day)')
            axes[1,2].set_ylabel('Frequency')
            axes[1,2].set_title('Precipitation Intensity Distribution')
            axes[1,2].grid(True, alpha=0.3)
        
        # Plot 7: Geographic distribution (county frequency heatmap simulation)
        county_problems = self.problem_records.groupby('GEOID').size()
        axes[2,0].hist(county_problems, bins=30, color='green', alpha=0.7, edgecolor='black')
        axes[2,0].set_xlabel('Problems per County')
        axes[2,0].set_ylabel('Number of Counties')
        axes[2,0].set_title('County Problem Frequency Distribution')
        axes[2,0].grid(True, alpha=0.3)
        
        # Plot 8: High precip days percentage of year
        axes[2,1].hist(self.problem_records['high_day_ratio'], bins=50, 
                      color='red', alpha=0.7, edgecolor='black')
        axes[2,1].set_xlabel('High Precip Days (% of year)')
        axes[2,1].set_ylabel('Frequency')
        axes[2,1].set_title('High Precip Days as Percentage of Year')
        axes[2,1].grid(True, alpha=0.3)
        
        # Plot 9: Comparison by scenario
        if len(self.problem_records['scenario'].unique()) > 1:
            sns.boxplot(data=self.problem_records, x='scenario', y='high_precip_days_95th', ax=axes[2,2])
            axes[2,2].set_title('High Precip Days by Scenario (Problem Records)')
            axes[2,2].set_ylabel('High Precipitation Days')
            axes[2,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'precipitation_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def investigate_specific_counties(self, top_n: int = 10):
        """Investigate specific counties with the most issues"""
        logger.info(f"Investigating top {top_n} counties with precipitation issues...")
        
        if self.problem_records is None or len(self.problem_records) == 0:
            logger.warning("No problem records to investigate")
            return
        
        top_problem_counties = self.problem_records['GEOID'].value_counts().head(top_n)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Top {top_n} Counties with Precipitation Issues', fontsize=16, fontweight='bold')
        
        # Plot 1: Time series for top problematic county
        top_county = top_problem_counties.index[0]
        county_data = self.df[self.df['GEOID'] == top_county].sort_values('year')
        county_problems = self.problem_records[self.problem_records['GEOID'] == top_county]
        
        axes[0,0].plot(county_data['year'], county_data['annual_precipitation_mm'], 
                      label='Annual Precipitation', linewidth=2)
        axes[0,0].scatter(county_problems['year'], county_problems['annual_precipitation_mm'], 
                         color='red', s=50, label='Problem Records', zorder=5)
        axes[0,0].set_xlabel('Year')
        axes[0,0].set_ylabel('Annual Precipitation (mm)')
        axes[0,0].set_title(f'County {top_county} - Annual Precipitation')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: High precip days time series for top county
        axes[0,1].plot(county_data['year'], county_data['high_precip_days_95th'], 
                      label='High Precip Days', linewidth=2, color='orange')
        axes[0,1].scatter(county_problems['year'], county_problems['high_precip_days_95th'], 
                         color='red', s=50, label='Problem Records', zorder=5)
        axes[0,1].set_xlabel('Year')
        axes[0,1].set_ylabel('High Precipitation Days')
        axes[0,1].set_title(f'County {top_county} - High Precipitation Days')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Comparison of normal vs problem records for top counties
        top_county_data = []
        for county in top_problem_counties.head(5).index:
            county_normal = self.df[(self.df['GEOID'] == county) & 
                                   (~self.df.index.isin(self.problem_records.index))]
            county_problem = self.problem_records[self.problem_records['GEOID'] == county]
            
            if len(county_normal) > 0 and len(county_problem) > 0:
                top_county_data.append({
                    'County': county,
                    'Normal_Avg_Precip': county_normal['annual_precipitation_mm'].mean(),
                    'Problem_Avg_Precip': county_problem['annual_precipitation_mm'].mean(),
                    'Normal_Avg_Days': county_normal['high_precip_days_95th'].mean(),
                    'Problem_Avg_Days': county_problem['high_precip_days_95th'].mean()
                })
        
        if top_county_data:
            comparison_df = pd.DataFrame(top_county_data)
            x = np.arange(len(comparison_df))
            width = 0.35
            
            axes[1,0].bar(x - width/2, comparison_df['Normal_Avg_Days'], width, 
                         label='Normal Records', alpha=0.7)
            axes[1,0].bar(x + width/2, comparison_df['Problem_Avg_Days'], width, 
                         label='Problem Records', alpha=0.7)
            axes[1,0].set_xlabel('County')
            axes[1,0].set_ylabel('Average High Precip Days')
            axes[1,0].set_title('Normal vs Problem Records - High Precip Days')
            axes[1,0].set_xticks(x)
            axes[1,0].set_xticklabels(comparison_df['County'], rotation=45)
            axes[1,0].legend()
        
        # Plot 4: Distribution of problem severity by county
        county_severity = self.problem_records.groupby('GEOID')['high_precip_days_95th'].max()
        axes[1,1].hist(county_severity, bins=20, color='red', alpha=0.7, edgecolor='black')
        axes[1,1].set_xlabel('Maximum High Precip Days per County')
        axes[1,1].set_ylabel('Number of Counties')
        axes[1,1].set_title('Distribution of Problem Severity by County')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'county_investigation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_detailed_report(self):
        """Generate detailed investigation report"""
        logger.info("Generating detailed investigation report...")
        
        analysis = self.analyze_problem_patterns()
        
        report_file = self.output_dir / f'precipitation_investigation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(report_file, 'w') as f:
            f.write("PRECIPITATION RELATIONSHIPS INVESTIGATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.data_file}\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total records analyzed: {len(self.df):,}\n")
            f.write(f"Problem records identified: {len(self.problem_records):,}\n")
            f.write(f"Problem rate: {len(self.problem_records)/len(self.df)*100:.2f}%\n\n")
            
            if analysis:
                f.write("PROBLEM PATTERNS\n")
                f.write("-" * 16 + "\n")
                f.write(f"Counties affected: {analysis['total_problem_counties']}\n")
                f.write(f"Extreme cases (>200 high precip days): {analysis['extreme_cases_count']}\n\n")
                
                f.write("TOP PROBLEM COUNTIES\n")
                f.write("-" * 20 + "\n")
                for county, count in list(analysis['top_problem_counties'].items())[:10]:
                    f.write(f"County {county}: {count} problem records\n")
                f.write("\n")
                
                f.write("PROBLEM DISTRIBUTION BY SCENARIO\n")
                f.write("-" * 33 + "\n")
                for scenario, count in analysis['problem_by_scenario'].items():
                    f.write(f"{scenario}: {count} records\n")
                f.write("\n")
                
                f.write("STATISTICAL SUMMARY OF PROBLEM RECORDS\n")
                f.write("-" * 39 + "\n")
                stats = analysis['precipitation_stats']
                f.write(f"Annual Precipitation Range: {stats['min_annual_precip']:.1f} - {stats['max_annual_precip']:.1f} mm\n")
                f.write(f"Annual Precipitation Mean: {stats['mean_annual_precip']:.1f} mm\n")
                f.write(f"High Precip Days Range: {stats['min_high_precip_days']:.1f} - {stats['max_high_precip_days']:.1f} days\n")
                f.write(f"High Precip Days Mean: {stats['mean_high_precip_days']:.1f} days\n\n")
            
            f.write("PROBLEM CATEGORIES IDENTIFIED\n")
            f.write("-" * 29 + "\n")
            f.write("1. Records with >100 high precipitation days (impossible)\n")
            f.write("2. Low annual precipitation (<100mm) but many high precip days (>20)\n")
            f.write("3. High precip days exceeding physical limits (>365 days)\n")
            f.write("4. Unrealistic ratios of high precip days to total precipitation\n\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            f.write("1. Investigate data processing pipeline for precipitation metrics\n")
            f.write("2. Review threshold calculations for high precipitation days\n")
            f.write("3. Validate source data for counties with persistent issues\n")
            f.write("4. Consider implementing automated QC checks for these patterns\n")
            f.write("5. Flag counties with >10 problem records for manual review\n\n")
            
            # Save sample of problem records
            if len(self.problem_records) > 0:
                f.write("SAMPLE PROBLEM RECORDS (First 20)\n")
                f.write("-" * 32 + "\n")
                sample_cols = ['GEOID', 'year', 'scenario', 'annual_precipitation_mm', 'high_precip_days_95th']
                sample_records = self.problem_records[sample_cols].head(20)
                f.write(sample_records.to_string(index=False))
                f.write("\n\n")
        
        logger.info(f"Detailed report saved to {report_file}")
        
        # Also save problem records as CSV for further analysis
        self.problem_records.to_csv(self.output_dir / 'problem_records.csv', index=False)
        logger.info(f"Problem records saved to {self.output_dir}/problem_records.csv")
    
    def run_full_investigation(self):
        """Run complete precipitation investigation"""
        logger.info("Starting comprehensive precipitation investigation...")
        
        # Load data
        self.load_data()
        
        # Identify issues
        self.identify_precipitation_issues()
        
        if self.problem_records is not None and len(self.problem_records) > 0:
            # Create diagnostic plots
            self.create_diagnostic_plots()
            
            # Investigate specific counties
            self.investigate_specific_counties()
            
            # Generate detailed report
            self.generate_detailed_report()
            
            logger.info(f"Investigation completed. Check {self.output_dir}/ for results.")
        else:
            logger.info("No precipitation relationship issues found.")


def main():
    """Main function to run precipitation investigation"""
    # Initialize investigator
    investigator = PrecipitationInvestigator("county_climate_metrics_complete_1980_2100.csv")
    
    # Run full investigation
    investigator.run_full_investigation()
    
    print("\nPRECIPITATION INVESTIGATION COMPLETE")
    print("=" * 45)
    print(f"Results saved to: {investigator.output_dir}/")
    print("Check the detailed report for findings and recommendations.")


if __name__ == "__main__":
    main() 