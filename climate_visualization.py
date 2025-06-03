#!/usr/bin/env python3
"""
Climate Data Visualization Suite
Comprehensive visualization for climate datasets organized by county and year
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
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import geopandas as gpd
import folium
from matplotlib.colors import LinearSegmentedColormap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('climate_visualization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set style for all plots
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class ClimateDataVisualizer:
    """
    Comprehensive visualization suite for climate datasets organized by county and year
    """
    
    def __init__(self, data_file: str, shapefile_path: str = "tl_2024_us_county/tl_2024_us_county.shp", output_dir: str = "climate_visualizations"):
        """Initialize visualizer with climate data file and county shapefile"""
        self.data_file = data_file
        self.shapefile_path = shapefile_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.df = None
        self.counties_gdf = None
        
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
            # CONUS state codes: 01-56 excluding AK(02), HI(15), and territories
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
    
    def create_climate_maps(self):
        """Create spatial maps of climate variables"""
        if self.counties_gdf is None:
            logger.warning("No county shapefile loaded, skipping map creation")
            return
            
        logger.info("Creating climate maps...")
        
        # Calculate average values for recent period (2015-2020)
        recent_data = self.df[
            (self.df['year'] >= 2015) & 
            (self.df['year'] <= 2020) & 
            (self.df['scenario'] == 'historical')
        ]
        
        if len(recent_data) == 0:
            # Fallback to any available historical data
            recent_data = self.df[self.df['scenario'] == 'historical'].tail(10000)
        
        # Calculate county averages for mapping
        county_climate = recent_data.groupby('GEOID').agg({
            'annual_mean_temp_c': 'mean',
            'annual_precipitation_mm': 'mean',
            'hot_days_30c': 'mean',
            'cold_days_0c': 'mean'
        }).reset_index()
        
        # Ensure GEOID is 5-character zero-padded string
        county_climate['GEOID'] = county_climate['GEOID'].astype(str).str.zfill(5)
        
        # Merge with shapefile
        map_data = self.counties_gdf.merge(county_climate, on='GEOID', how='left')
        
        # Create maps
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Map 1: Average Annual Temperature
        if 'annual_mean_temp_c' in map_data.columns:
            map_data.plot(column='annual_mean_temp_c', ax=axes[0,0], legend=True,
                         cmap='RdYlBu_r', edgecolor='black', linewidth=0.1,
                         legend_kwds={'label': 'Temperature (°C)'})
            axes[0,0].set_title('Average Annual Temperature (2015-2020)', fontweight='bold', fontsize=14)
            axes[0,0].axis('off')
        
        # Map 2: Average Annual Precipitation
        if 'annual_precipitation_mm' in map_data.columns:
            map_data.plot(column='annual_precipitation_mm', ax=axes[0,1], legend=True,
                         cmap='Blues', edgecolor='black', linewidth=0.1,
                         legend_kwds={'label': 'Precipitation (mm)'})
            axes[0,1].set_title('Average Annual Precipitation (2015-2020)', fontweight='bold', fontsize=14)
            axes[0,1].axis('off')
        
        # Map 3: Hot Days (>30°C)
        if 'hot_days_30c' in map_data.columns:
            map_data.plot(column='hot_days_30c', ax=axes[1,0], legend=True,
                         cmap='Reds', edgecolor='black', linewidth=0.1,
                         legend_kwds={'label': 'Days'})
            axes[1,0].set_title('Average Hot Days >30°C (2015-2020)', fontweight='bold', fontsize=14)
            axes[1,0].axis('off')
        
        # Map 4: Cold Days (<0°C)
        if 'cold_days_0c' in map_data.columns:
            map_data.plot(column='cold_days_0c', ax=axes[1,1], legend=True,
                         cmap='Blues_r', edgecolor='black', linewidth=0.1,
                         legend_kwds={'label': 'Days'})
            axes[1,1].set_title('Average Cold Days <0°C (2015-2020)', fontweight='bold', fontsize=14)
            axes[1,1].axis('off')
        
        plt.suptitle('Climate Maps - CONUS Counties', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'climate_maps.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create climate change maps (future vs historical)
        self._create_climate_change_maps()
    
    def _create_climate_change_maps(self):
        """Create maps showing climate change projections"""
        logger.info("Creating climate change projection maps...")
        
        # Historical average (1980-2014)
        historical_data = self.df[
            (self.df['year'] >= 1980) & 
            (self.df['year'] <= 2014) & 
            (self.df['scenario'] == 'historical')
        ].groupby('GEOID').agg({
            'annual_mean_temp_c': 'mean',
            'annual_precipitation_mm': 'mean'
        }).reset_index()
        historical_data.columns = ['GEOID', 'hist_temp', 'hist_precip']
        
        # Future projection average (2050-2080)
        future_data = self.df[
            (self.df['year'] >= 2050) & 
            (self.df['year'] <= 2080) & 
            (self.df['scenario'] == 'ssp245')
        ].groupby('GEOID').agg({
            'annual_mean_temp_c': 'mean',
            'annual_precipitation_mm': 'mean'
        }).reset_index()
        future_data.columns = ['GEOID', 'future_temp', 'future_precip']
        
        # Calculate change
        change_data = historical_data.merge(future_data, on='GEOID', how='inner')
        change_data['temp_change'] = change_data['future_temp'] - change_data['hist_temp']
        change_data['precip_change_pct'] = ((change_data['future_precip'] - change_data['hist_precip']) / 
                                           change_data['hist_precip'] * 100)
        
        # Ensure GEOID is 5-character zero-padded string
        change_data['GEOID'] = change_data['GEOID'].astype(str).str.zfill(5)
        
        # Merge with shapefile
        map_data = self.counties_gdf.merge(change_data, on='GEOID', how='left')
        
        # Create change maps
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Temperature change map
        if 'temp_change' in map_data.columns:
            vmin, vmax = map_data['temp_change'].quantile([0.05, 0.95])
            map_data.plot(column='temp_change', ax=axes[0], legend=True,
                         cmap='RdBu_r', edgecolor='black', linewidth=0.1,
                         vmin=vmin, vmax=vmax,
                         legend_kwds={'label': 'Temperature Change (°C)'})
            axes[0].set_title('Projected Temperature Change\n(2050-2080 vs 1980-2014)', 
                             fontweight='bold', fontsize=14)
            axes[0].axis('off')
        
        # Precipitation change map
        if 'precip_change_pct' in map_data.columns:
            vmin, vmax = map_data['precip_change_pct'].quantile([0.05, 0.95])
            map_data.plot(column='precip_change_pct', ax=axes[1], legend=True,
                         cmap='BrBG', edgecolor='black', linewidth=0.1,
                         vmin=vmin, vmax=vmax,
                         legend_kwds={'label': 'Precipitation Change (%)'})
            axes[1].set_title('Projected Precipitation Change\n(2050-2080 vs 1980-2014)', 
                             fontweight='bold', fontsize=14)
            axes[1].axis('off')
        
        plt.suptitle('Climate Change Projections (SSP2-4.5)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'climate_change_maps.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_data_coverage_map(self):
        """Create map showing data coverage and quality"""
        if self.counties_gdf is None:
            return
            
        logger.info("Creating data coverage map...")
        
        # Calculate data coverage metrics
        coverage_data = self.df.groupby('GEOID').agg({
            'year': ['count', 'min', 'max'],
            'scenario': 'nunique'
        }).reset_index()
        
        coverage_data.columns = ['GEOID', 'total_records', 'min_year', 'max_year', 'scenarios']
        coverage_data['year_span'] = coverage_data['max_year'] - coverage_data['min_year']
        
        # Ensure GEOID is 5-character zero-padded string
        coverage_data['GEOID'] = coverage_data['GEOID'].astype(str).str.zfill(5)
        
        # Merge with shapefile
        map_data = self.counties_gdf.merge(coverage_data, on='GEOID', how='left')
        
        # Create coverage maps
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Total records map
        map_data.plot(column='total_records', ax=axes[0], legend=True,
                     cmap='Greens', edgecolor='black', linewidth=0.1,
                     legend_kwds={'label': 'Number of Records'})
        axes[0].set_title('Total Climate Records per County', fontweight='bold', fontsize=14)
        axes[0].axis('off')
        
        # Year span map
        map_data.plot(column='year_span', ax=axes[1], legend=True,
                     cmap='Blues', edgecolor='black', linewidth=0.1,
                     legend_kwds={'label': 'Years'})
        axes[1].set_title('Temporal Coverage (Year Span)', fontweight='bold', fontsize=14)
        axes[1].axis('off')
        
        # Scenarios map
        map_data.plot(column='scenarios', ax=axes[2], legend=True,
                     cmap='Oranges', edgecolor='black', linewidth=0.1,
                     legend_kwds={'label': 'Number of Scenarios'})
        axes[2].set_title('Climate Scenarios Available', fontweight='bold', fontsize=14)
        axes[2].axis('off')
        
        plt.suptitle('Data Coverage and Quality Assessment', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'data_coverage_map.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_overview_dashboard(self):
        """Create comprehensive overview dashboard"""
        logger.info("Creating overview dashboard...")
        
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: Data completeness by year and scenario
        ax1 = fig.add_subplot(gs[0, :2])
        completeness = self.df.groupby(['year', 'scenario']).size().unstack(fill_value=0)
        completeness.plot(kind='line', marker='o', ax=ax1, linewidth=2, markersize=4)
        ax1.set_title('Data Completeness by Year and Scenario', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Counties')
        ax1.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: County distribution by scenario
        ax2 = fig.add_subplot(gs[0, 2:])
        scenario_counts = self.df['scenario'].value_counts()
        colors = sns.color_palette("Set2", len(scenario_counts))
        ax2.pie(scenario_counts.values, labels=scenario_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax2.set_title('Data Distribution by Scenario', fontsize=14, fontweight='bold')
        
        # Plot 3: Temperature trends
        if 'annual_mean_temp_c' in self.df.columns:
            ax3 = fig.add_subplot(gs[1, :2])
            temp_trends = self.df.groupby(['year', 'scenario'])['annual_mean_temp_c'].mean().unstack()
            temp_trends.plot(kind='line', marker='o', ax=ax3, linewidth=2, markersize=3)
            ax3.set_title('Average Annual Temperature Trends', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Year')
            ax3.set_ylabel('Temperature (°C)')
            ax3.legend(title='Scenario')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Precipitation trends
        if 'annual_precipitation_mm' in self.df.columns:
            ax4 = fig.add_subplot(gs[1, 2:])
            precip_trends = self.df.groupby(['year', 'scenario'])['annual_precipitation_mm'].mean().unstack()
            precip_trends.plot(kind='line', marker='o', ax=ax4, linewidth=2, markersize=3)
            ax4.set_title('Average Annual Precipitation Trends', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Precipitation (mm)')
            ax4.legend(title='Scenario')
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Temperature distribution by scenario
        if 'annual_mean_temp_c' in self.df.columns:
            ax5 = fig.add_subplot(gs[2, :2])
            sns.boxplot(data=self.df, x='scenario', y='annual_mean_temp_c', ax=ax5)
            ax5.set_title('Temperature Distribution by Scenario', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Annual Mean Temperature (°C)')
            
        # Plot 6: Precipitation distribution by scenario
        if 'annual_precipitation_mm' in self.df.columns:
            ax6 = fig.add_subplot(gs[2, 2:])
            sns.boxplot(data=self.df, x='scenario', y='annual_precipitation_mm', ax=ax6)
            ax6.set_title('Precipitation Distribution by Scenario', fontsize=14, fontweight='bold')
            ax6.set_ylabel('Annual Precipitation (mm)')
        
        # Plot 7: Missing data heatmap
        ax7 = fig.add_subplot(gs[3, :2])
        missing_data = self.df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            missing_data.plot(kind='bar', ax=ax7, color='salmon')
            ax7.set_title('Missing Data by Variable', fontsize=14, fontweight='bold')
            ax7.set_ylabel('Number of Missing Values')
            ax7.tick_params(axis='x', rotation=45)
        else:
            ax7.text(0.5, 0.5, 'No Missing Data', transform=ax7.transAxes, 
                    ha='center', va='center', fontsize=16)
            ax7.set_title('Missing Data Status', fontsize=14, fontweight='bold')
        
        # Plot 8: Data coverage timeline
        ax8 = fig.add_subplot(gs[3, 2:])
        yearly_counts = self.df.groupby('year').size()
        ax8.fill_between(yearly_counts.index, yearly_counts.values, alpha=0.7, color='skyblue')
        ax8.plot(yearly_counts.index, yearly_counts.values, color='navy', linewidth=2)
        ax8.set_title('Data Coverage Over Time', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Year')
        ax8.set_ylabel('Number of Records')
        ax8.grid(True, alpha=0.3)
        
        plt.suptitle('Climate Data Overview Dashboard', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'overview_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_temperature_analysis(self):
        """Create detailed temperature analysis plots"""
        logger.info("Creating temperature analysis plots...")
        
        temp_columns = [col for col in self.df.columns if 'temp' in col.lower()]
        if not temp_columns:
            logger.warning("No temperature columns found")
            return
        
        # Temperature correlation matrix
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Correlation heatmap
        temp_corr = self.df[temp_columns].corr()
        sns.heatmap(temp_corr, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[0,0], fmt='.2f', square=True)
        axes[0,0].set_title('Temperature Variables Correlation', fontweight='bold')
        
        # Temperature range analysis
        if all(col in self.df.columns for col in ['annual_min_temp_c', 'annual_max_temp_c']):
            self.df['temp_range'] = self.df['annual_max_temp_c'] - self.df['annual_min_temp_c']
            sns.boxplot(data=self.df, x='scenario', y='temp_range', ax=axes[0,1])
            axes[0,1].set_title('Temperature Range by Scenario', fontweight='bold')
            axes[0,1].set_ylabel('Temperature Range (°C)')
        
        # Extreme temperature days analysis
        extreme_temp_cols = [col for col in self.df.columns if any(x in col.lower() for x in ['hot_days', 'cold_days'])]
        if extreme_temp_cols and len(extreme_temp_cols) >= 2:
            # Use first two extreme temperature columns
            col1, col2 = extreme_temp_cols[:2]
            sns.scatterplot(data=self.df.sample(n=min(5000, len(self.df))), 
                           x=col1, y=col2, hue='scenario', alpha=0.6, ax=axes[1,0])
            axes[1,0].set_title(f'{col1} vs {col2}', fontweight='bold')
        
        # Temperature trends by decade
        if 'annual_mean_temp_c' in self.df.columns:
            self.df['decade'] = (self.df['year'] // 10) * 10
            decade_temps = self.df.groupby(['decade', 'scenario'])['annual_mean_temp_c'].mean().unstack()
            decade_temps.plot(kind='bar', ax=axes[1,1], width=0.8)
            axes[1,1].set_title('Average Temperature by Decade', fontweight='bold')
            axes[1,1].set_ylabel('Temperature (°C)')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].legend(title='Scenario')
        
        plt.suptitle('Temperature Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temperature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_precipitation_analysis(self):
        """Create detailed precipitation analysis plots"""
        logger.info("Creating precipitation analysis plots...")
        
        precip_columns = [col for col in self.df.columns if 'precip' in col.lower()]
        if not precip_columns:
            logger.warning("No precipitation columns found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Precipitation distribution
        if 'annual_precipitation_mm' in self.df.columns:
            for scenario in self.df['scenario'].unique():
                scenario_data = self.df[self.df['scenario'] == scenario]['annual_precipitation_mm']
                axes[0,0].hist(scenario_data, bins=50, alpha=0.7, label=scenario, density=True)
            axes[0,0].set_title('Precipitation Distribution by Scenario', fontweight='bold')
            axes[0,0].set_xlabel('Annual Precipitation (mm)')
            axes[0,0].set_ylabel('Density')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        
        # Precipitation vs high precipitation days
        if all(col in self.df.columns for col in ['annual_precipitation_mm', 'high_precip_days_95th']):
            sample_data = self.df.sample(n=min(10000, len(self.df)))
            sns.scatterplot(data=sample_data, x='annual_precipitation_mm', y='high_precip_days_95th', 
                           hue='scenario', alpha=0.6, ax=axes[0,1])
            axes[0,1].set_title('Precipitation vs High Precipitation Days', fontweight='bold')
            axes[0,1].set_xlabel('Annual Precipitation (mm)')
            axes[0,1].set_ylabel('High Precipitation Days')
        
        # Precipitation trends
        if 'annual_precipitation_mm' in self.df.columns:
            precip_trends = self.df.groupby(['year', 'scenario'])['annual_precipitation_mm'].mean().unstack()
            precip_trends.plot(kind='line', marker='o', ax=axes[1,0], linewidth=2, markersize=3)
            axes[1,0].set_title('Precipitation Trends Over Time', fontweight='bold')
            axes[1,0].set_xlabel('Year')
            axes[1,0].set_ylabel('Average Precipitation (mm)')
            axes[1,0].legend(title='Scenario')
            axes[1,0].grid(True, alpha=0.3)
        
        # Seasonal precipitation analysis (if available)
        seasonal_cols = [col for col in precip_columns if any(season in col.lower() for season in ['spring', 'summer', 'fall', 'winter'])]
        if seasonal_cols:
            seasonal_data = self.df[seasonal_cols + ['scenario']].groupby('scenario').mean()
            seasonal_data.T.plot(kind='bar', ax=axes[1,1])
            axes[1,1].set_title('Seasonal Precipitation Patterns', fontweight='bold')
            axes[1,1].set_ylabel('Precipitation (mm)')
            axes[1,1].tick_params(axis='x', rotation=45)
        else:
            # Precipitation extremes
            if 'annual_precipitation_mm' in self.df.columns:
                extreme_precip = self.df.groupby('scenario')['annual_precipitation_mm'].agg(['min', 'max', 'std'])
                extreme_precip.plot(kind='bar', ax=axes[1,1])
                axes[1,1].set_title('Precipitation Extremes by Scenario', fontweight='bold')
                axes[1,1].set_ylabel('Precipitation (mm)')
                axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Precipitation Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'precipitation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_extreme_events_analysis(self):
        """Create extreme weather events analysis"""
        logger.info("Creating extreme events analysis...")
        
        extreme_cols = [col for col in self.df.columns if any(word in col.lower() 
                       for word in ['extreme', 'hot_days', 'cold_days', 'high_temp', 'low_temp'])]
        
        if not extreme_cols:
            logger.warning("No extreme event columns found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Hot vs cold days
        hot_cols = [col for col in extreme_cols if 'hot' in col.lower()]
        cold_cols = [col for col in extreme_cols if 'cold' in col.lower()]
        
        if hot_cols and cold_cols:
            sample_data = self.df.sample(n=min(5000, len(self.df)))
            sns.scatterplot(data=sample_data, x=hot_cols[0], y=cold_cols[0], 
                           hue='scenario', alpha=0.6, ax=axes[0,0])
            axes[0,0].set_title(f'{hot_cols[0]} vs {cold_cols[0]}', fontweight='bold')
        
        # Extreme events trends
        if extreme_cols:
            extreme_trends = self.df.groupby(['year', 'scenario'])[extreme_cols[:3]].mean()
            for i, col in enumerate(extreme_cols[:3]):
                scenario_trends = extreme_trends[col].unstack()
                if i == 0:
                    ax = axes[0,1]
                elif i == 1:
                    ax = axes[1,0]
                else:
                    ax = axes[1,1]
                
                scenario_trends.plot(kind='line', marker='o', ax=ax, linewidth=2, markersize=2)
                ax.set_title(f'{col} Trends', fontweight='bold')
                ax.set_xlabel('Year')
                ax.set_ylabel('Days')
                ax.legend(title='Scenario')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Extreme Weather Events Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'extreme_events_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_spatial_analysis(self):
        """Create spatial analysis plots"""
        logger.info("Creating spatial analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # County data availability
        county_counts = self.df.groupby('GEOID').size()
        axes[0,0].hist(county_counts, bins=50, color='skyblue', alpha=0.7)
        axes[0,0].set_title('Records per County Distribution', fontweight='bold')
        axes[0,0].set_xlabel('Number of Records')
        axes[0,0].set_ylabel('Number of Counties')
        axes[0,0].grid(True, alpha=0.3)
        
        # Top counties with most data
        top_counties = county_counts.nlargest(20)
        top_counties.plot(kind='bar', ax=axes[0,1], color='lightcoral')
        axes[0,1].set_title('Top 20 Counties by Data Volume', fontweight='bold')
        axes[0,1].set_xlabel('County GEOID')
        axes[0,1].set_ylabel('Number of Records')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Temperature variation by county (sample)
        if 'annual_mean_temp_c' in self.df.columns:
            county_temp_var = self.df.groupby('GEOID')['annual_mean_temp_c'].std().dropna()
            axes[1,0].hist(county_temp_var, bins=50, color='orange', alpha=0.7)
            axes[1,0].set_title('Temperature Variability by County', fontweight='bold')
            axes[1,0].set_xlabel('Temperature Standard Deviation (°C)')
            axes[1,0].set_ylabel('Number of Counties')
            axes[1,0].grid(True, alpha=0.3)
        
        # Precipitation variation by county (sample)
        if 'annual_precipitation_mm' in self.df.columns:
            county_precip_var = self.df.groupby('GEOID')['annual_precipitation_mm'].std().dropna()
            axes[1,1].hist(county_precip_var, bins=50, color='lightgreen', alpha=0.7)
            axes[1,1].set_title('Precipitation Variability by County', fontweight='bold')
            axes[1,1].set_xlabel('Precipitation Standard Deviation (mm)')
            axes[1,1].set_ylabel('Number of Counties')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle('Spatial Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'spatial_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_scenario_comparison(self):
        """Create detailed scenario comparison plots"""
        logger.info("Creating scenario comparison plots...")
        
        scenarios = self.df['scenario'].unique()
        if len(scenarios) < 2:
            logger.warning("Need at least 2 scenarios for comparison")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Climate metrics for comparison
        metrics = []
        if 'annual_mean_temp_c' in self.df.columns:
            metrics.append('annual_mean_temp_c')
        if 'annual_precipitation_mm' in self.df.columns:
            metrics.append('annual_precipitation_mm')
        if 'growing_degree_days' in self.df.columns:
            metrics.append('growing_degree_days')
        
        for i, metric in enumerate(metrics[:6]):
            row = i // 3
            col = i % 3
            
            # Box plot comparison
            sns.boxplot(data=self.df, x='scenario', y=metric, ax=axes[row, col])
            axes[row, col].set_title(f'{metric} by Scenario', fontweight='bold')
            axes[row, col].tick_params(axis='x', rotation=45)
        
        # Fill remaining subplots if needed
        for i in range(len(metrics), 6):
            row = i // 3
            col = i % 3
            axes[row, col].axis('off')
        
        plt.suptitle('Climate Scenarios Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scenario_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_temporal_patterns(self):
        """Create temporal pattern analysis"""
        logger.info("Creating temporal pattern analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Yearly data volume
        yearly_counts = self.df.groupby(['year', 'scenario']).size().unstack(fill_value=0)
        yearly_counts.plot(kind='area', ax=axes[0,0], alpha=0.7, stacked=True)
        axes[0,0].set_title('Data Volume Over Time by Scenario', fontweight='bold')
        axes[0,0].set_xlabel('Year')
        axes[0,0].set_ylabel('Number of Records')
        axes[0,0].legend(title='Scenario')
        
        # Decadal analysis
        if 'annual_mean_temp_c' in self.df.columns:
            self.df['decade'] = (self.df['year'] // 10) * 10
            decade_analysis = self.df.groupby(['decade', 'scenario']).agg({
                'annual_mean_temp_c': 'mean',
                'GEOID': 'count'
            }).reset_index()
            
            decade_pivot = decade_analysis.pivot(index='decade', columns='scenario', values='annual_mean_temp_c')
            decade_pivot.plot(kind='line', marker='o', ax=axes[0,1], linewidth=3, markersize=6)
            axes[0,1].set_title('Decadal Temperature Trends', fontweight='bold')
            axes[0,1].set_xlabel('Decade')
            axes[0,1].set_ylabel('Average Temperature (°C)')
            axes[0,1].legend(title='Scenario')
            axes[0,1].grid(True, alpha=0.3)
        
        # Data gaps analysis
        year_range = range(self.df['year'].min(), self.df['year'].max() + 1)
        scenario_year_combinations = [(s, y) for s in self.df['scenario'].unique() for y in year_range]
        actual_combinations = set(self.df[['scenario', 'year']].itertuples(index=False, name=None))
        missing_combinations = [combo for combo in scenario_year_combinations if combo not in actual_combinations]
        
        if missing_combinations:
            missing_df = pd.DataFrame(missing_combinations, columns=['scenario', 'year'])
            missing_counts = missing_df.groupby(['year', 'scenario']).size().unstack(fill_value=0)
            missing_counts.plot(kind='bar', ax=axes[1,0], stacked=True, width=0.8)
            axes[1,0].set_title('Missing Data Patterns Over Time', fontweight='bold')
            axes[1,0].set_xlabel('Year')
            axes[1,0].set_ylabel('Missing Scenario-Year Combinations')
            axes[1,0].legend(title='Scenario')
            axes[1,0].tick_params(axis='x', rotation=45)
        else:
            axes[1,0].text(0.5, 0.5, 'No Missing Data Patterns', 
                          transform=axes[1,0].transAxes, ha='center', va='center', fontsize=14)
            axes[1,0].set_title('Data Completeness Status', fontweight='bold')
        
        # Recent vs historical comparison
        if 'annual_mean_temp_c' in self.df.columns:
            historical_data = self.df[self.df['year'] <= 2014]
            future_data = self.df[self.df['year'] > 2014]
            
            if len(historical_data) > 0 and len(future_data) > 0:
                hist_temp = historical_data.groupby('scenario')['annual_mean_temp_c'].mean()
                future_temp = future_data.groupby('scenario')['annual_mean_temp_c'].mean()
                
                comparison_data = pd.DataFrame({
                    'Historical (≤2014)': hist_temp,
                    'Future (>2014)': future_temp
                }).fillna(0)
                
                comparison_data.plot(kind='bar', ax=axes[1,1], width=0.8)
                axes[1,1].set_title('Historical vs Future Temperature Comparison', fontweight='bold')
                axes[1,1].set_xlabel('Scenario')
                axes[1,1].set_ylabel('Average Temperature (°C)')
                axes[1,1].tick_params(axis='x', rotation=45)
                axes[1,1].legend()
        
        plt.suptitle('Temporal Patterns Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_summary_report(self):
        """Generate a summary visualization report"""
        logger.info("Generating summary report...")
        
        # Calculate key statistics
        stats = {
            'total_records': len(self.df),
            'unique_counties': self.df['GEOID'].nunique(),
            'year_range': f"{self.df['year'].min()}-{self.df['year'].max()}",
            'scenarios': list(self.df['scenario'].unique()),
            'missing_data_pct': (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        }
        
        # Save summary report
        with open(self.output_dir / 'visualization_summary.txt', 'w') as f:
            f.write("CLIMATE DATA VISUALIZATION SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.data_file}\n\n")
            
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Records: {stats['total_records']:,}\n")
            f.write(f"Unique Counties: {stats['unique_counties']:,}\n")
            f.write(f"Year Range: {stats['year_range']}\n")
            f.write(f"Scenarios: {', '.join(stats['scenarios'])}\n")
            f.write(f"Missing Data: {stats['missing_data_pct']:.2f}%\n\n")
            
            f.write("GENERATED VISUALIZATIONS\n")
            f.write("-" * 25 + "\n")
            f.write("1. overview_dashboard.png - Comprehensive overview\n")
            f.write("2. temperature_analysis.png - Temperature patterns\n")
            f.write("3. precipitation_analysis.png - Precipitation patterns\n")
            f.write("4. extreme_events_analysis.png - Extreme weather events\n")
            f.write("5. spatial_analysis.png - Spatial distribution patterns\n")
            f.write("6. scenario_comparison.png - Climate scenario comparisons\n")
            f.write("7. temporal_patterns.png - Time-based patterns\n")
            
            # Add spatial maps if available
            if self.counties_gdf is not None:
                f.write("8. climate_maps.png - Spatial maps of current climate\n")
                f.write("9. climate_change_maps.png - Climate change projection maps\n")
                f.write("10. data_coverage_map.png - Data coverage quality maps\n\n")
                
                f.write("SPATIAL MAPPING FEATURES\n")
                f.write("-" * 25 + "\n")
                f.write("- County-level choropleth maps for CONUS\n")
                f.write("- Current climate conditions (2015-2020 average)\n")
                f.write("- Climate change projections (2050-2080 vs 1980-2014)\n")
                f.write("- Data coverage and quality assessment\n")
                f.write("- Temperature, precipitation, and extreme events mapping\n")
            else:
                f.write("\nNote: Spatial maps not generated (shapefile not available)\n")
        
        logger.info(f"Summary report saved to {self.output_dir}/visualization_summary.txt")
    
    def create_all_visualizations(self):
        """Create all visualization plots"""
        logger.info("Starting comprehensive visualization generation...")
        
        # Load data
        self.load_data()
        
        # Load shapefile for maps
        self.load_counties_shapefile()
        
        # Create all visualizations
        self.create_overview_dashboard()
        self.create_temperature_analysis()
        self.create_precipitation_analysis()
        self.create_extreme_events_analysis()
        self.create_spatial_analysis()
        self.create_scenario_comparison()
        self.create_temporal_patterns()
        
        # Create spatial maps
        if self.counties_gdf is not None:
            self.create_climate_maps()
            self.create_data_coverage_map()
        
        # Generate summary report
        self.generate_summary_report()
        
        logger.info(f"All visualizations completed. Check {self.output_dir}/ for outputs.")


def main():
    """Main function to run climate data visualization"""
    # Initialize visualizer
    visualizer = ClimateDataVisualizer("county_climate_metrics_complete_1980_2100.csv")
    
    # Create all visualizations
    visualizer.create_all_visualizations()
    
    print("\nVISUALIZATION COMPLETE")
    print("=" * 40)
    print(f"All plots saved to: {visualizer.output_dir}/")
    print("Check visualization_summary.txt for detailed information.")


if __name__ == "__main__":
    main() 