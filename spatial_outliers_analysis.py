#!/usr/bin/env python3
"""
Spatial Outliers Analysis
Focused examination of counties with consistently extreme climate values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import logging
from datetime import datetime
from scipy import stats
import geopandas as gpd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spatial_outliers_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class SpatialOutliersAnalyzer:
    """
    Analyze counties with consistently extreme climate values
    """
    
    def __init__(self, data_file: str, shapefile_path: str = "tl_2024_us_county/tl_2024_us_county.shp", output_dir: str = "spatial_outliers_analysis"):
        """Initialize analyzer with climate data file and county shapefile"""
        self.data_file = data_file
        self.shapefile_path = shapefile_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.df = None
        self.counties_gdf = None
        self.outlier_counties = {}
        self.climate_metrics = []
        
    def load_data(self) -> pd.DataFrame:
        """Load climate data"""
        logger.info(f"Loading climate data from {self.data_file}")
        
        try:
            self.df = pd.read_csv(self.data_file)
            logger.info(f"Loaded {len(self.df):,} records with {len(self.df.columns)} columns")
            
            # Identify climate metrics columns
            exclude_cols = ['GEOID', 'year', 'scenario', 'pixel_count']
            self.climate_metrics = [col for col in self.df.columns 
                                  if col not in exclude_cols and self.df[col].dtype in ['int64', 'float64']]
            logger.info(f"Identified {len(self.climate_metrics)} climate metrics for analysis")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def load_counties_shapefile(self) -> gpd.GeoDataFrame:
        """Load county boundaries shapefile"""
        logger.info(f"Loading county shapefile from {self.shapefile_path}")
        
        try:
            self.counties_gdf = gpd.read_file(self.shapefile_path)
            logger.info(f"Loaded shapefile with {len(self.counties_gdf)} counties")
            
            # Ensure GEOID is string and properly formatted (5-character zero-padded)
            self.counties_gdf['GEOID'] = self.counties_gdf['GEOID'].astype(str).str.zfill(5)
            
            # Filter to continental US (exclude Alaska, Hawaii, territories)
            exclude_states = ['02', '15', '60', '66', '69', '72', '78']
            self.counties_gdf = self.counties_gdf[
                ~self.counties_gdf['GEOID'].str[:2].isin(exclude_states)
            ]
            
            logger.info(f"Filtered to {len(self.counties_gdf)} CONUS counties")
            return self.counties_gdf
            
        except Exception as e:
            logger.error(f"Failed to load shapefile: {str(e)}")
            logger.warning("Map visualizations will be skipped")
            return None
    
    def identify_spatial_outliers(self, outlier_method: str = 'iqr', threshold_multiplier: float = 3.0):
        """Identify spatial outliers using specified method"""
        logger.info(f"Identifying spatial outliers using {outlier_method} method...")
        
        outlier_counties = {}
        
        for metric in self.climate_metrics:
            metric_outliers = []
            
            # For each year-scenario combination, find outliers
            for (year, scenario), group in self.df.groupby(['year', 'scenario']):
                if len(group) < 10:  # Skip if too few counties
                    continue
                
                values = group[metric].dropna()
                if len(values) < 10:
                    continue
                
                if outlier_method == 'iqr':
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold_multiplier * IQR
                    upper_bound = Q3 + threshold_multiplier * IQR
                    
                    outlier_mask = (values < lower_bound) | (values > upper_bound)
                    
                elif outlier_method == 'zscore':
                    z_scores = np.abs(stats.zscore(values))
                    outlier_mask = z_scores > threshold_multiplier
                    
                elif outlier_method == 'modified_zscore':
                    median = values.median()
                    mad = np.median(np.abs(values - median))
                    modified_z_scores = 0.6745 * (values - median) / mad
                    outlier_mask = np.abs(modified_z_scores) > threshold_multiplier
                
                # Get outlier counties for this year-scenario
                if outlier_mask.any():
                    outlier_values = values[outlier_mask]
                    outlier_geoids = group.loc[outlier_values.index, 'GEOID'].tolist()
                    metric_outliers.extend(outlier_geoids)
            
            if metric_outliers:
                # Count frequency of outliers per county
                outlier_freq = pd.Series(metric_outliers).value_counts()
                outlier_counties[metric] = outlier_freq.head(20).to_dict()  # Top 20 outlier counties
        
        self.outlier_counties = outlier_counties
        logger.info(f"Found outliers in {len(outlier_counties)} climate metrics")
        
        return outlier_counties
    
    def analyze_persistent_outliers(self):
        """Analyze counties that are persistently outliers across multiple metrics"""
        logger.info("Analyzing persistent outliers...")
        
        if not self.outlier_counties:
            logger.warning("No outliers identified yet")
            return
        
        # Count how many metrics each county appears as an outlier in
        county_outlier_counts = {}
        county_metric_details = {}
        
        for metric, county_counts in self.outlier_counties.items():
            for county, count in county_counts.items():
                if county not in county_outlier_counts:
                    county_outlier_counts[county] = 0
                    county_metric_details[county] = {}
                
                county_outlier_counts[county] += 1
                county_metric_details[county][metric] = count
        
        # Create persistent outliers dataframe
        persistent_outliers = []
        for county, metric_count in county_outlier_counts.items():
            total_outlier_occurrences = sum(county_metric_details[county].values())
            persistent_outliers.append({
                'GEOID': county,
                'metrics_as_outlier': metric_count,
                'total_outlier_occurrences': total_outlier_occurrences,
                'outlier_metrics': list(county_metric_details[county].keys())
            })
        
        self.persistent_outliers_df = pd.DataFrame(persistent_outliers).sort_values(
            'metrics_as_outlier', ascending=False
        )
        
        logger.info(f"Found {len(self.persistent_outliers_df)} counties with outlier patterns")
        
        return self.persistent_outliers_df
    
    def investigate_top_outlier_counties(self, top_n: int = 10):
        """Detailed investigation of top outlier counties"""
        logger.info(f"Investigating top {top_n} outlier counties...")
        
        if not hasattr(self, 'persistent_outliers_df'):
            self.analyze_persistent_outliers()
        
        top_counties = self.persistent_outliers_df.head(top_n)['GEOID'].tolist()
        county_investigations = {}
        
        for county in top_counties:
            county_data = self.df[self.df['GEOID'] == county]
            investigation = {
                'total_records': len(county_data),
                'scenarios': county_data['scenario'].unique().tolist(),
                'year_range': f"{county_data['year'].min()}-{county_data['year'].max()}",
                'climate_summary': {}
            }
            
            # Calculate statistics for each climate metric
            for metric in self.climate_metrics:
                if metric in county_data.columns:
                    metric_data = county_data[metric].dropna()
                    if len(metric_data) > 0:
                        # Compare with overall dataset
                        overall_mean = self.df[metric].mean()
                        overall_std = self.df[metric].std()
                        county_mean = metric_data.mean()
                        
                        investigation['climate_summary'][metric] = {
                            'county_mean': county_mean,
                            'overall_mean': overall_mean,
                            'deviation_from_mean': county_mean - overall_mean,
                            'z_score': (county_mean - overall_mean) / overall_std if overall_std > 0 else 0,
                            'county_std': metric_data.std(),
                            'min_value': metric_data.min(),
                            'max_value': metric_data.max()
                        }
            
            county_investigations[county] = investigation
        
        self.county_investigations = county_investigations
        return county_investigations
    
    def create_spatial_outliers_dashboard(self):
        """Create comprehensive dashboard for spatial outliers"""
        logger.info("Creating spatial outliers dashboard...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Spatial Outliers Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: Number of outlier metrics per county
        if hasattr(self, 'persistent_outliers_df'):
            top_persistent = self.persistent_outliers_df.head(20)
            axes[0,0].barh(range(len(top_persistent)), top_persistent['metrics_as_outlier'], 
                          color='coral')
            axes[0,0].set_yticks(range(len(top_persistent)))
            axes[0,0].set_yticklabels(top_persistent['GEOID'])
            axes[0,0].set_xlabel('Number of Metrics as Outlier')
            axes[0,0].set_title('Counties with Most Outlier Metrics')
            axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Distribution of outlier occurrences
        if hasattr(self, 'persistent_outliers_df'):
            axes[0,1].hist(self.persistent_outliers_df['total_outlier_occurrences'], 
                          bins=30, color='skyblue', alpha=0.7, edgecolor='black')
            axes[0,1].set_xlabel('Total Outlier Occurrences')
            axes[0,1].set_ylabel('Number of Counties')
            axes[0,1].set_title('Distribution of Outlier Frequency')
            axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Outliers by metric
        metric_outlier_counts = {}
        for metric, counties in self.outlier_counties.items():
            metric_outlier_counts[metric] = len(counties)
        
        if metric_outlier_counts:
            metrics = list(metric_outlier_counts.keys())[:15]  # Top 15 metrics
            counts = [metric_outlier_counts[m] for m in metrics]
            axes[0,2].bar(range(len(metrics)), counts, color='lightgreen')
            axes[0,2].set_xticks(range(len(metrics)))
            axes[0,2].set_xticklabels(metrics, rotation=45, ha='right')
            axes[0,2].set_ylabel('Number of Outlier Counties')
            axes[0,2].set_title('Outlier Counties by Climate Metric')
        
        # Plot 4-6: Top outlier counties detailed analysis
        if hasattr(self, 'county_investigations'):
            top_3_counties = list(self.county_investigations.keys())[:3]
            
            for i, county in enumerate(top_3_counties):
                ax = axes[1, i]
                investigation = self.county_investigations[county]
                
                # Plot z-scores for this county
                metrics = []
                z_scores = []
                for metric, stats in investigation['climate_summary'].items():
                    metrics.append(metric[:20])  # Truncate long names
                    z_scores.append(stats['z_score'])
                
                if metrics:
                    colors = ['red' if abs(z) > 2 else 'orange' if abs(z) > 1 else 'green' 
                             for z in z_scores]
                    ax.barh(range(len(metrics)), z_scores, color=colors, alpha=0.7)
                    ax.set_yticks(range(len(metrics)))
                    ax.set_yticklabels(metrics, fontsize=8)
                    ax.set_xlabel('Z-Score')
                    ax.set_title(f'County {county} - Climate Deviations')
                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    ax.axvline(x=2, color='red', linestyle='--', alpha=0.5)
                    ax.axvline(x=-2, color='red', linestyle='--', alpha=0.5)
                    ax.grid(True, alpha=0.3)
        
        # Plot 7: Geographic clustering analysis
        if hasattr(self, 'persistent_outliers_df'):
            # Analyze GEOID patterns to identify geographic clustering
            geoids = self.persistent_outliers_df['GEOID'].astype(str)
            state_codes = geoids.str[:2].astype(int)
            state_outlier_counts = state_codes.value_counts().head(15)
            
            axes[2,0].bar(range(len(state_outlier_counts)), state_outlier_counts.values, 
                         color='purple', alpha=0.7)
            axes[2,0].set_xticks(range(len(state_outlier_counts)))
            axes[2,0].set_xticklabels(state_outlier_counts.index, rotation=45)
            axes[2,0].set_xlabel('State Code (First 2 digits of GEOID)')
            axes[2,0].set_ylabel('Number of Outlier Counties')
            axes[2,0].set_title('Outlier Counties by State')
        
        # Plot 8: Temporal patterns in outliers
        outlier_temporal_analysis = {}
        for metric, counties in self.outlier_counties.items():
            for county in counties.keys():
                county_data = self.df[self.df['GEOID'] == county]
                for scenario in county_data['scenario'].unique():
                    scenario_data = county_data[county_data['scenario'] == scenario]
                    key = f"{metric}_{scenario}"
                    if key not in outlier_temporal_analysis:
                        outlier_temporal_analysis[key] = []
                    outlier_temporal_analysis[key].extend(scenario_data['year'].tolist())
        
        # Show year distribution for outliers
        all_outlier_years = []
        for years_list in outlier_temporal_analysis.values():
            all_outlier_years.extend(years_list)
        
        if all_outlier_years:
            axes[2,1].hist(all_outlier_years, bins=30, color='orange', alpha=0.7, edgecolor='black')
            axes[2,1].set_xlabel('Year')
            axes[2,1].set_ylabel('Frequency of Outlier Occurrences')
            axes[2,1].set_title('Temporal Distribution of Outliers')
            axes[2,1].grid(True, alpha=0.3)
        
        # Plot 9: Outlier severity distribution
        all_z_scores = []
        if hasattr(self, 'county_investigations'):
            for investigation in self.county_investigations.values():
                for stats in investigation['climate_summary'].values():
                    all_z_scores.append(abs(stats['z_score']))
        
        if all_z_scores:
            axes[2,2].hist(all_z_scores, bins=30, color='red', alpha=0.7, edgecolor='black')
            axes[2,2].set_xlabel('Absolute Z-Score')
            axes[2,2].set_ylabel('Frequency')
            axes[2,2].set_title('Distribution of Outlier Severity')
            axes[2,2].axvline(x=2, color='black', linestyle='--', label='2σ threshold')
            axes[2,2].axvline(x=3, color='red', linestyle='--', label='3σ threshold')
            axes[2,2].legend()
            axes[2,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'spatial_outliers_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_county_profiles(self, top_n: int = 5):
        """Create detailed profiles for top outlier counties"""
        logger.info(f"Creating detailed profiles for top {top_n} outlier counties...")
        
        if not hasattr(self, 'county_investigations'):
            self.investigate_top_outlier_counties(top_n)
        
        top_counties = list(self.county_investigations.keys())[:top_n]
        
        for county in top_counties:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'County {county} - Detailed Climate Profile', fontsize=16, fontweight='bold')
            
            county_data = self.df[self.df['GEOID'] == county]
            investigation = self.county_investigations[county]
            
            # Plot 1: Temperature metrics over time
            temp_metrics = [col for col in self.climate_metrics if 'temp' in col.lower()]
            if temp_metrics:
                for metric in temp_metrics[:3]:  # Top 3 temperature metrics
                    for scenario in county_data['scenario'].unique():
                        scenario_data = county_data[county_data['scenario'] == scenario].sort_values('year')
                        if len(scenario_data) > 0 and metric in scenario_data.columns:
                            axes[0,0].plot(scenario_data['year'], scenario_data[metric], 
                                         marker='o', label=f"{metric}_{scenario}", linewidth=2, markersize=3)
                
                axes[0,0].set_xlabel('Year')
                axes[0,0].set_ylabel('Temperature (°C)')
                axes[0,0].set_title('Temperature Metrics Over Time')
                axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[0,0].grid(True, alpha=0.3)
            
            # Plot 2: Precipitation metrics over time
            precip_metrics = [col for col in self.climate_metrics if 'precip' in col.lower()]
            if precip_metrics:
                for metric in precip_metrics[:2]:  # Top 2 precipitation metrics
                    for scenario in county_data['scenario'].unique():
                        scenario_data = county_data[county_data['scenario'] == scenario].sort_values('year')
                        if len(scenario_data) > 0 and metric in scenario_data.columns:
                            axes[0,1].plot(scenario_data['year'], scenario_data[metric], 
                                         marker='o', label=f"{metric}_{scenario}", linewidth=2, markersize=3)
                
                axes[0,1].set_xlabel('Year')
                axes[0,1].set_ylabel('Precipitation')
                axes[0,1].set_title('Precipitation Metrics Over Time')
                axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[0,1].grid(True, alpha=0.3)
            
            # Plot 3: Z-score comparison
            metrics = []
            z_scores = []
            for metric, stats in investigation['climate_summary'].items():
                if abs(stats['z_score']) > 0.5:  # Only show significant deviations
                    metrics.append(metric[:15])  # Truncate names
                    z_scores.append(stats['z_score'])
            
            if metrics:
                colors = ['red' if abs(z) > 2 else 'orange' if abs(z) > 1 else 'green' for z in z_scores]
                axes[0,2].barh(range(len(metrics)), z_scores, color=colors, alpha=0.7)
                axes[0,2].set_yticks(range(len(metrics)))
                axes[0,2].set_yticklabels(metrics, fontsize=10)
                axes[0,2].set_xlabel('Z-Score (Deviation from National Mean)')
                axes[0,2].set_title('Climate Metric Deviations')
                axes[0,2].axvline(x=0, color='black', linestyle='-', alpha=0.3)
                axes[0,2].axvline(x=2, color='red', linestyle='--', alpha=0.5, label='±2σ')
                axes[0,2].axvline(x=-2, color='red', linestyle='--', alpha=0.5)
                axes[0,2].legend()
                axes[0,2].grid(True, alpha=0.3)
            
            # Plot 4: Extreme events over time
            extreme_metrics = [col for col in self.climate_metrics 
                             if any(word in col.lower() for word in ['hot_days', 'cold_days', 'extreme'])]
            if extreme_metrics:
                for metric in extreme_metrics[:2]:  # Top 2 extreme metrics
                    for scenario in county_data['scenario'].unique():
                        scenario_data = county_data[county_data['scenario'] == scenario].sort_values('year')
                        if len(scenario_data) > 0 and metric in scenario_data.columns:
                            axes[1,0].plot(scenario_data['year'], scenario_data[metric], 
                                         marker='o', label=f"{metric}_{scenario}", linewidth=2, markersize=3)
                
                axes[1,0].set_xlabel('Year')
                axes[1,0].set_ylabel('Days')
                axes[1,0].set_title('Extreme Weather Events Over Time')
                axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[1,0].grid(True, alpha=0.3)
            
            # Plot 5: County vs National distribution comparison
            if 'annual_mean_temp_c' in county_data.columns:
                national_temp = self.df['annual_mean_temp_c'].dropna()
                county_temp = county_data['annual_mean_temp_c'].dropna()
                
                axes[1,1].hist(national_temp, bins=50, alpha=0.5, label='National', 
                              density=True, color='lightblue')
                axes[1,1].hist(county_temp, bins=20, alpha=0.7, label=f'County {county}', 
                              density=True, color='red')
                axes[1,1].set_xlabel('Annual Mean Temperature (°C)')
                axes[1,1].set_ylabel('Density')
                axes[1,1].set_title('Temperature Distribution Comparison')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
            
            # Plot 6: Variability analysis
            metrics_variability = {}
            for metric in self.climate_metrics[:10]:  # Top 10 metrics
                if metric in county_data.columns:
                    county_std = county_data[metric].std()
                    national_std = self.df[metric].std()
                    if national_std > 0:
                        metrics_variability[metric[:15]] = county_std / national_std
            
            if metrics_variability:
                metrics = list(metrics_variability.keys())
                ratios = list(metrics_variability.values())
                colors = ['red' if r > 1.5 else 'orange' if r > 1.2 else 'green' for r in ratios]
                
                axes[1,2].bar(range(len(metrics)), ratios, color=colors, alpha=0.7)
                axes[1,2].set_xticks(range(len(metrics)))
                axes[1,2].set_xticklabels(metrics, rotation=45, ha='right')
                axes[1,2].set_ylabel('Variability Ratio (County/National)')
                axes[1,2].set_title('Climate Variability Comparison')
                axes[1,2].axhline(y=1, color='black', linestyle='--', alpha=0.5)
                axes[1,2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'county_{county}_profile.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_spatial_outliers_maps(self):
        """Create maps showing spatial distribution of outliers"""
        if self.counties_gdf is None:
            logger.warning("No county shapefile loaded, skipping map creation")
            return
            
        logger.info("Creating spatial outliers maps...")
        
        if not hasattr(self, 'persistent_outliers_df'):
            self.analyze_persistent_outliers()
        
        # Ensure GEOID is 5-character zero-padded string
        self.persistent_outliers_df['GEOID'] = self.persistent_outliers_df['GEOID'].astype(str).str.zfill(5)
        
        # Merge outlier data with shapefile
        map_data = self.counties_gdf.merge(
            self.persistent_outliers_df[['GEOID', 'metrics_as_outlier', 'total_outlier_occurrences']], 
            on='GEOID', how='left'
        )
        
        # Fill NaN values for counties with no outliers
        map_data['metrics_as_outlier'] = map_data['metrics_as_outlier'].fillna(0)
        map_data['total_outlier_occurrences'] = map_data['total_outlier_occurrences'].fillna(0)
        
        # Create outlier maps
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Map 1: Number of metrics as outlier
        map_data.plot(column='metrics_as_outlier', ax=axes[0], legend=True,
                     cmap='Reds', edgecolor='black', linewidth=0.1,
                     legend_kwds={'label': 'Number of Outlier Metrics'})
        axes[0].set_title('Counties by Number of Outlier Climate Metrics', fontweight='bold', fontsize=14)
        axes[0].axis('off')
        
        # Map 2: Total outlier occurrences
        map_data.plot(column='total_outlier_occurrences', ax=axes[1], legend=True,
                     cmap='Oranges', edgecolor='black', linewidth=0.1,
                     legend_kwds={'label': 'Total Outlier Occurrences'})
        axes[1].set_title('Counties by Total Outlier Occurrences', fontweight='bold', fontsize=14)
        axes[1].axis('off')
        
        plt.suptitle('Spatial Distribution of Climate Outliers', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'spatial_outliers_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create metric-specific outlier maps
        self._create_metric_specific_maps()
    
    def _create_metric_specific_maps(self):
        """Create maps for specific climate metrics with highest outlier counts"""
        logger.info("Creating metric-specific outlier maps...")
        
        # Get top 4 metrics with most outliers
        metric_outlier_counts = {metric: len(counties) for metric, counties in self.outlier_counties.items()}
        top_metrics = sorted(metric_outlier_counts.items(), key=lambda x: x[1], reverse=True)[:4]
        
        if len(top_metrics) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, (metric, _) in enumerate(top_metrics):
            # Create outlier indicator for this metric
            outlier_geoids = set(self.outlier_counties[metric].keys())
            
            # Ensure outlier GEOIDs are 5-character zero-padded strings
            outlier_geoids = {str(geoid).zfill(5) for geoid in outlier_geoids}
            
            # Create map data
            map_data = self.counties_gdf.copy()
            map_data['is_outlier'] = map_data['GEOID'].isin(outlier_geoids).astype(int)
            
            # Plot map
            map_data.plot(column='is_outlier', ax=axes[i], legend=True,
                         cmap='RdYlBu_r', edgecolor='black', linewidth=0.1,
                         legend_kwds={'label': 'Outlier Status'})
            
            axes[i].set_title(f'{metric}\nOutlier Counties', fontweight='bold', fontsize=12)
            axes[i].axis('off')
        
        plt.suptitle('Climate Metric Specific Outlier Maps', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metric_specific_outlier_maps.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_extreme_counties_map(self):
        """Create map highlighting extreme outlier counties"""
        if self.counties_gdf is None:
            return
            
        logger.info("Creating extreme counties map...")
        
        if not hasattr(self, 'persistent_outliers_df'):
            self.analyze_persistent_outliers()
        
        # Ensure GEOID is 5-character zero-padded string
        self.persistent_outliers_df['GEOID'] = self.persistent_outliers_df['GEOID'].astype(str).str.zfill(5)
        
        # Define extreme outlier counties (>5 metrics as outliers)
        extreme_counties = self.persistent_outliers_df[
            self.persistent_outliers_df['metrics_as_outlier'] > 5
        ]['GEOID'].tolist()
        
        # Create map data
        map_data = self.counties_gdf.copy()
        map_data['outlier_category'] = 'Normal'
        
        # Categorize counties
        persistent_counties = self.persistent_outliers_df[
            (self.persistent_outliers_df['metrics_as_outlier'] > 1) & 
            (self.persistent_outliers_df['metrics_as_outlier'] <= 5)
        ]['GEOID'].tolist()
        
        map_data.loc[map_data['GEOID'].isin(persistent_counties), 'outlier_category'] = 'Moderate Outlier'
        map_data.loc[map_data['GEOID'].isin(extreme_counties), 'outlier_category'] = 'Extreme Outlier'
        
        # Create categorical color map
        category_colors = {'Normal': 'lightgray', 'Moderate Outlier': 'orange', 'Extreme Outlier': 'red'}
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        for category, color in category_colors.items():
            subset = map_data[map_data['outlier_category'] == category]
            if len(subset) > 0:
                subset.plot(ax=ax, color=color, edgecolor='black', linewidth=0.1, label=category)
        
        ax.set_title('Climate Outlier Counties Classification', fontweight='bold', fontsize=16)
        ax.axis('off')
        ax.legend(loc='lower right', fontsize=12)
        
        # Add text annotation
        ax.text(0.02, 0.98, f'Extreme Outliers: {len(extreme_counties)} counties\n'
                            f'Moderate Outliers: {len(persistent_counties)} counties\n'
                            f'Normal: {len(map_data) - len(extreme_counties) - len(persistent_counties)} counties',
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'extreme_counties_map.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self):
        """Generate comprehensive spatial outliers report"""
        logger.info("Generating comprehensive spatial outliers report...")
        
        report_file = self.output_dir / f'spatial_outliers_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(report_file, 'w') as f:
            f.write("SPATIAL OUTLIERS ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.data_file}\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total records analyzed: {len(self.df):,}\n")
            f.write(f"Unique counties: {self.df['GEOID'].nunique():,}\n")
            f.write(f"Climate metrics analyzed: {len(self.climate_metrics)}\n")
            
            if hasattr(self, 'persistent_outliers_df'):
                f.write(f"Counties with outlier patterns: {len(self.persistent_outliers_df)}\n")
                f.write(f"Counties outliers in >5 metrics: {len(self.persistent_outliers_df[self.persistent_outliers_df['metrics_as_outlier'] > 5])}\n\n")
            
            f.write("OUTLIER DETECTION SUMMARY\n")
            f.write("-" * 26 + "\n")
            for metric, counties in self.outlier_counties.items():
                f.write(f"{metric}: {len(counties)} outlier counties\n")
            f.write("\n")
            
            if hasattr(self, 'persistent_outliers_df'):
                f.write("TOP PERSISTENT OUTLIER COUNTIES\n")
                f.write("-" * 32 + "\n")
                top_10 = self.persistent_outliers_df.head(10)
                for _, row in top_10.iterrows():
                    f.write(f"County {row['GEOID']}: Outlier in {row['metrics_as_outlier']} metrics "
                           f"({row['total_outlier_occurrences']} total occurrences)\n")
                f.write("\n")
            
            if hasattr(self, 'county_investigations'):
                f.write("DETAILED COUNTY INVESTIGATIONS\n")
                f.write("-" * 31 + "\n")
                for county, investigation in list(self.county_investigations.items())[:5]:
                    f.write(f"\nCounty {county}:\n")
                    f.write(f"  Records: {investigation['total_records']}\n")
                    f.write(f"  Scenarios: {', '.join(investigation['scenarios'])}\n")
                    f.write(f"  Years: {investigation['year_range']}\n")
                    f.write("  Most extreme deviations:\n")
                    
                    # Sort by absolute z-score
                    extreme_deviations = []
                    for metric, stats in investigation['climate_summary'].items():
                        extreme_deviations.append((metric, abs(stats['z_score']), stats['z_score']))
                    
                    extreme_deviations.sort(key=lambda x: x[1], reverse=True)
                    for metric, abs_z, z_score in extreme_deviations[:5]:
                        f.write(f"    {metric}: {z_score:.2f}σ\n")
            
            f.write("\nCLIMATE METRICS ANALYZED\n")
            f.write("-" * 24 + "\n")
            for i, metric in enumerate(self.climate_metrics, 1):
                f.write(f"{i}. {metric}\n")
            
            f.write("\nMETHODOLOGY\n")
            f.write("-" * 11 + "\n")
            f.write("Outlier detection method: IQR with 3.0x threshold\n")
            f.write("Applied separately to each year-scenario combination\n")
            f.write("Counties flagged if consistently outliers across time periods\n")
            f.write("Z-scores calculated relative to full dataset mean and std\n\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            f.write("1. Investigate counties with >5 outlier metrics for data quality issues\n")
            f.write("2. Review geographic/topographic factors for persistent outliers\n")
            f.write("3. Validate source data for counties with extreme z-scores (>3σ)\n")
            f.write("4. Consider regional climate patterns when interpreting outliers\n")
            f.write("5. Flag top outlier counties for manual review in future analyses\n")
        
        logger.info(f"Comprehensive report saved to {report_file}")
        
        # Save outlier counties data
        if hasattr(self, 'persistent_outliers_df'):
            self.persistent_outliers_df.to_csv(self.output_dir / 'persistent_outlier_counties.csv', index=False)
            logger.info(f"Persistent outlier counties data saved to {self.output_dir}/persistent_outlier_counties.csv")
    
    def run_full_analysis(self):
        """Run complete spatial outliers analysis"""
        logger.info("Starting comprehensive spatial outliers analysis...")
        
        # Load data
        self.load_data()
        
        # Load shapefile for maps
        self.load_counties_shapefile()
        
        # Identify outliers
        self.identify_spatial_outliers()
        
        # Analyze persistent outliers
        self.analyze_persistent_outliers()
        
        # Investigate top counties
        self.investigate_top_outlier_counties()
        
        # Create visualizations
        self.create_spatial_outliers_dashboard()
        self.create_county_profiles()
        
        # Create spatial maps
        if self.counties_gdf is not None:
            self.create_spatial_outliers_maps()
            self.create_extreme_counties_map()
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        logger.info(f"Analysis completed. Check {self.output_dir}/ for results.")


def main():
    """Main function to run spatial outliers analysis"""
    # Initialize analyzer
    analyzer = SpatialOutliersAnalyzer("county_climate_metrics_complete_1980_2100.csv")
    
    # Run full analysis
    analyzer.run_full_analysis()
    
    print("\nSPATIAL OUTLIERS ANALYSIS COMPLETE")
    print("=" * 40)
    print(f"Results saved to: {analyzer.output_dir}/")
    print("Check the comprehensive report for detailed findings.")


if __name__ == "__main__":
    main() 