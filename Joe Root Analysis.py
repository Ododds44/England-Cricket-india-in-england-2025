import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import joblib
import warnings
import re
from datetime import datetime
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class JoeRootAnalyzer:
    """A comprehensive analyzer for Joe Root's Test performance against India"""
    
    def __init__(self, data_path=None):
        """Initialize the analyzer with dataset"""
        self.data = None
        self.run_predictor = None
        self.win_predictor = None
        self.label_encoders = {}
        self.team_performance = {}
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """Load the dataset from CSV file (ESPNCricinfo format)"""
        self.data = pd.read_csv(data_path)
        print(f"Data loaded: {self.data.shape[0]} rows")
        print(f"Columns: {list(self.data.columns)}")
    
    def load_team_performance_data(self):
        """Load England vs India team performance statistics"""
        # Overall historical stats
        self.team_performance['overall'] = {
            'total_matches': 131,
            'england_wins': 50,
            'india_wins': 31,
            'draws': 50,
            'england_win_rate': 50/131,
            'india_win_rate': 31/131,
            'draw_rate': 50/131
        }
        
        # Home/Away specific performance
        self.team_performance['england_home'] = {
            'england_win_rate': 0.45,
            'india_win_rate': 0.25,
            'draw_rate': 0.30
        }
        
        self.team_performance['england_away'] = {
            'england_win_rate': 0.20,
            'india_win_rate': 0.50,
            'draw_rate': 0.30
        }
        
        # Recent form adjustment
        self.team_performance['recent_form'] = {
            'england_home': {'england_win_rate': 0.40, 'india_win_rate': 0.35, 'draw_rate': 0.25},
            'england_away': {'england_win_rate': 0.15, 'india_win_rate': 0.60, 'draw_rate': 0.25}
        }
        
        print("Team performance data loaded")
        print(f"Historical: England {self.team_performance['overall']['england_wins']} wins, "
              f"India {self.team_performance['overall']['india_wins']} wins, "
              f"{self.team_performance['overall']['draws']} draws")
    
    def calculate_team_based_win_probability(self, venue, root_contribution_factor=0.4):
        """Calculate win probability based on team performance and venue"""
        england_venues = ['Headingley', 'Edgbaston', "Lord's", 'Old Trafford', 
                         'The Oval', 'Trent Bridge', 'Southampton']
        
        is_home = venue in england_venues
        
        if is_home:
            base_win_prob = self.team_performance['recent_form']['england_home']['england_win_rate']
        else:
            base_win_prob = self.team_performance['recent_form']['england_away']['england_win_rate']
        
        return base_win_prob
    
    def parse_espn_data(self):
        """Parse ESPNCricinfo CSV format to extract required information"""
        processed_data = []
        
        for idx, row in self.data.iterrows():
            # Skip if no batting data
            if pd.isna(row.get('Bat1')) or row.get('Bat1') == '-':
                continue
                
            # Extract match info
            match_date = row.get('Start Date', '')
            ground = row.get('Ground', '')
            
            # Parse batting innings (Bat1 and Bat2)
            for innings_num, bat_col in enumerate(['Bat1', 'Bat2'], 1):
                if pd.isna(row.get(bat_col)) or row.get(bat_col) == '-':
                    continue
                    
                bat_info = str(row[bat_col])
                
                # Skip non-batting entries
                if bat_info in ['DNB', 'TDNB', 'absent', '-']:
                    continue
                
                # Extract runs using regex
                runs_match = re.search(r'(\d+)', bat_info)
                if not runs_match:
                    continue
                    
                runs = int(runs_match.group(1))
                not_out = '*' in bat_info
                
                # Determine match result
                result = row.get('Result', '')
                match_result = self._parse_match_result(result)
                
                # Create innings record
                innings_data = {
                    'Match date': match_date,
                    'Ground': ground,
                    'Opposition': 'India',
                    'Innings number': innings_num,
                    'Runs': runs,
                    'Not Out': not_out,
                    'Result': match_result,
                    'Batting position': 4,
                    'Match_ID': idx
                }
                
                processed_data.append(innings_data)
        
        self.data = pd.DataFrame(processed_data)
        
        # Clean venue names
        self.data['Ground_Clean'] = self.data['Ground'].apply(self._clean_venue_name)
        
        print(f"\nProcessed {len(self.data)} innings")
        print(f"Venues found: {self.data['Ground_Clean'].unique()}")
        
        # Add additional features
        self._add_match_features()
        
        return self.data
    
    def _parse_match_result(self, result):
        """Parse match result string"""
        if pd.isna(result):
            return 'Unknown'
            
        result_lower = result.lower()
        
        if 'won' in result_lower:
            if 'England' in result:
                return 'Won'
            else:
                return 'Lost'
        elif 'drawn' in result_lower or 'draw' in result_lower:
            return 'Drawn'
        elif 'lost' in result_lower:
            return 'Lost'
        else:
            return 'Unknown'
    
    def _clean_venue_name(self, venue):
        """Extract main venue name from full ground name"""
        if pd.isna(venue):
            return 'Unknown'
        
        venue = str(venue)
        
        # England venues
        if 'Lord' in venue:
            return "Lord's"
        elif 'Headingley' in venue:
            return 'Headingley'
        elif 'Edgbaston' in venue:
            return 'Edgbaston'
        elif 'Old Trafford' in venue:
            return 'Old Trafford'
        elif 'Oval' in venue:
            return 'The Oval'
        elif 'Trent Bridge' in venue:
            return 'Trent Bridge'
        elif 'Southampton' in venue or 'Rose Bowl' in venue:
            return 'Southampton'
        
        # Indian venues
        elif 'Mumbai' in venue or 'Wankhede' in venue:
            return 'Mumbai'
        elif 'Delhi' in venue or 'Feroz Shah' in venue:
            return 'Delhi'
        elif 'Kolkata' in venue or 'Eden Gardens' in venue:
            return 'Kolkata'
        elif 'Chennai' in venue or 'Chepauk' in venue:
            return 'Chennai'
        elif 'Bangalore' in venue or 'Bengaluru' in venue or 'Chinnaswamy' in venue:
            return 'Bangalore'
        elif 'Mohali' in venue:
            return 'Mohali'
        elif 'Nagpur' in venue:
            return 'Nagpur'
        elif 'Rajkot' in venue:
            return 'Rajkot'
        elif 'Visakhapatnam' in venue or 'Vizag' in venue:
            return 'Visakhapatnam'
        elif 'Ahmedabad' in venue:
            return 'Ahmedabad'
        elif 'Dharamsala' in venue:
            return 'Dharamsala'
        else:
            return venue.split(',')[0].strip()
    
    def _add_match_features(self):
        """Add toss and batting first features based on innings patterns"""
        self.data = self.data.sort_values(['Match_ID', 'Innings number'])
        self.data['Batting first indicator'] = (self.data['Innings number'] <= 2).astype(int)
        self.data['Toss result'] = self.data['Batting first indicator'].map({1: 'won', 0: 'lost'})
    
    def prepare_data(self):
        """Part 1: Data Preparation"""
        if 'Ground_Clean' not in self.data.columns:
            self.data['Ground_Clean'] = self.data['Ground']
        
        self.data['Ground'] = self.data['Ground_Clean']
        
        # Ensure Runs is numeric
        self.data['Runs'] = pd.to_numeric(self.data['Runs'], errors='coerce')
        self.data = self.data.dropna(subset=['Runs'])
        
        # Convert Match date to datetime
        self.data['Match date'] = pd.to_datetime(self.data['Match date'], errors='coerce')
        self.data = self.data.sort_values('Match date')
        
        # Calculate form_last_5
        self.data['form_last_5'] = self.data['Runs'].shift(1).rolling(window=5, min_periods=1).mean()
        self.data['form_last_5'] = self.data['form_last_5'].fillna(self.data['Runs'].mean())
        
        # Extract year
        self.data['Year'] = self.data['Match date'].dt.year
        self.data['Year'] = self.data['Year'].fillna(2020)
        
        # Create binary win column
        self.data['Win'] = (self.data['Result'] == 'Won').astype(int)
        
        # Encode categorical variables
        categorical_cols = ['Ground', 'Toss result']
        for col in categorical_cols:
            if col in self.data.columns:
                le = LabelEncoder()
                self.data[f'{col}_encoded'] = le.fit_transform(self.data[col])
                self.label_encoders[col] = le
        
        # Add innings type
        self.data['Is_First_Innings'] = (self.data['Innings number'] <= 2).astype(int)
        
        print("Data preparation completed")
        print(f"Final dataset shape: {self.data.shape}")
        print(f"Date range: {self.data['Match date'].min()} to {self.data['Match date'].max()}")
        print(f"Venues included: {sorted(self.data['Ground'].unique())}")
        
        return self.data
    
    def train_run_prediction_model(self):
        """Part 2: Train regression model to predict Joe Root's runs"""
        feature_cols = ['Ground_encoded', 'Innings number', 'Year', 
                       'Toss result_encoded', 'Batting first indicator', 'form_last_5']
        
        available_features = [col for col in feature_cols if col in self.data.columns]
        
        X = self.data[available_features]
        y = self.data['Runs']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.run_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.run_predictor.fit(X_train, y_train)
        
        y_pred = self.run_predictor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\nRun Prediction Model Performance:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.3f}")
        print(f"Average prediction error: {np.sqrt(mse):.1f} runs")
        
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': self.run_predictor.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance for Run Prediction:")
        print(feature_importance)
        
        return self.run_predictor
    
    def train_win_probability_model(self):
        """Part 3: Train classification model to predict England win probability"""
        # Load team performance data
        self.load_team_performance_data()
        
        feature_cols = ['Ground_encoded', 'Innings number', 'Year', 
                       'Toss result_encoded', 'Batting first indicator', 'form_last_5']
        available_features = [col for col in feature_cols if col in self.data.columns]
        
        self.data['predicted_runs'] = self.run_predictor.predict(self.data[available_features])
        
        # Add team performance features
        england_venues = ['Headingley', 'Edgbaston', "Lord's", 'Old Trafford', 
                         'The Oval', 'Trent Bridge', 'Southampton']
        
        self.data['Is_Home'] = self.data['Ground'].isin(england_venues).astype(int)
        self.data['Base_Win_Prob'] = self.data['Is_Home'].apply(
            lambda x: self.team_performance['recent_form']['england_home']['england_win_rate'] 
            if x else self.team_performance['recent_form']['england_away']['england_win_rate']
        )
        
        win_features = ['predicted_runs', 'Ground_encoded', 'Innings number', 
                       'Is_Home', 'Base_Win_Prob']
        
        X = self.data[win_features]
        y = self.data['Win']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.win_predictor = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.win_predictor.fit(X_train, y_train)
        
        y_pred = self.win_predictor.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\nEnhanced Win Probability Model Performance:")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Loss/Draw', 'Win']))
        
        feature_importance = pd.DataFrame({
            'feature': win_features,
            'importance': self.win_predictor.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance for Win Prediction:")
        print(feature_importance)
        
        return self.win_predictor
    
    def predict_root_runs(self, venue, innings_number, form_last_5, 
                         toss_result='won', batting_first=1, year=2024):
        """Function to predict Joe Root's runs for given conditions"""
        if venue not in self.label_encoders['Ground'].classes_:
            raise ValueError(f"Unknown venue: {venue}")
        
        venue_encoded = self.label_encoders['Ground'].transform([venue])[0]
        
        if toss_result not in self.label_encoders['Toss result'].classes_:
            toss_result = 'won'
        
        toss_encoded = self.label_encoders['Toss result'].transform([toss_result])[0]
        
        features = pd.DataFrame({
            'Ground_encoded': [venue_encoded],
            'Innings number': [innings_number],
            'Year': [year],
            'Toss result_encoded': [toss_encoded],
            'Batting first indicator': [batting_first],
            'form_last_5': [form_last_5]
        })
        
        predicted_runs = self.run_predictor.predict(features)[0]
        
        return predicted_runs
    
    def win_probability(self, root_runs, venue, innings_number):
        """Enhanced function to calculate England's win probability"""
        if venue not in self.label_encoders['Ground'].classes_:
            venue_encoded = self.data['Ground_encoded'].mode()[0]
        else:
            venue_encoded = self.label_encoders['Ground'].transform([venue])[0]
        
        england_venues = ['Headingley', 'Edgbaston', "Lord's", 'Old Trafford', 
                         'The Oval', 'Trent Bridge', 'Southampton']
        is_home = 1 if venue in england_venues else 0
        
        base_win_prob = self.calculate_team_based_win_probability(venue)
        
        features = pd.DataFrame({
            'predicted_runs': [root_runs],
            'Ground_encoded': [venue_encoded],
            'Innings number': [innings_number],
            'Is_Home': [is_home],
            'Base_Win_Prob': [base_win_prob]
        })
        
        win_prob = self.win_predictor.predict_proba(features)[0, 1]
        
        root_avg = self.data['Runs'].mean()
        root_impact = (root_runs - root_avg) / root_avg
        
        adjustment = root_impact * 0.2
        adjusted_prob = win_prob + adjustment
        
        adjusted_prob = max(0.05, min(0.95, adjusted_prob))
        
        return adjusted_prob
    
    def save_models(self, prefix='joe_root'):
        """Save trained models using joblib"""
        joblib.dump(self.run_predictor, f'{prefix}_run_predictor.pkl')
        joblib.dump(self.win_predictor, f'{prefix}_win_predictor.pkl')
        joblib.dump(self.label_encoders, f'{prefix}_label_encoders.pkl')
        print(f"\nModels saved with prefix: {prefix}")
    
    def create_visualizations(self):
        """Part 4: Create all required visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Line chart: Root's runs over time at 5 venues
        ax1 = axes[0, 0]
        for venue in self.data['Ground'].unique():
            venue_data = self.data[self.data['Ground'] == venue].sort_values('Match date')
            ax1.plot(venue_data['Match date'], venue_data['Runs'], 
                    marker='o', label=venue, alpha=0.7)
        
        ax1.set_xlabel('Match Date')
        ax1.set_ylabel('Runs Scored')
        ax1.set_title("Joe Root's Runs Over Time by Venue")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Bar chart: Average runs per venue
        ax2 = axes[0, 1]
        avg_runs = self.data.groupby('Ground')['Runs'].agg(['mean', 'std'])
        venues = avg_runs.index
        means = avg_runs['mean'].values
        stds = avg_runs['std'].values
        
        bars = ax2.bar(venues, means, yerr=stds, capsize=5, alpha=0.7)
        ax2.set_xlabel('Venue')
        ax2.set_ylabel('Average Runs')
        ax2.set_title('Average Runs per Venue (with std dev)')
        ax2.set_xticklabels(venues, rotation=45, ha='right')
        
        for bar, mean in zip(bars, means):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{mean:.1f}', ha='center', va='bottom')
        
        # 3. Heatmap: Average runs by venue and innings
        ax3 = axes[1, 0]
        pivot_table = self.data.pivot_table(
            values='Runs', 
            index='Ground', 
            columns='Innings number', 
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Average Runs'}, ax=ax3)
        ax3.set_title('Average Runs by Venue and Innings')
        ax3.set_xlabel('Innings Number')
        ax3.set_ylabel('Venue')
        
        # 4. Logistic curve: Win probability vs Root's runs
        ax4 = axes[1, 1]
        runs_range = np.linspace(0, 200, 100)
        
        win_probs = []
        for runs in runs_range:
            avg_ground = self.data['Ground_encoded'].mode()[0]
            avg_innings = self.data['Innings number'].mode()[0]
            
            X_temp = pd.DataFrame({
                'predicted_runs': [runs],
                'Ground_encoded': [avg_ground],
                'Innings number': [avg_innings],
                'Is_Home': [1],
                'Base_Win_Prob': [0.4]
            })
            
            prob = self.win_predictor.predict_proba(X_temp)[0, 1]
            win_probs.append(prob)
        
        ax4.plot(runs_range, win_probs, linewidth=2.5, color='darkblue')
        ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% probability')
        ax4.set_xlabel("Joe Root's Runs")
        ax4.set_ylabel('England Win Probability')
        ax4.set_title('Win Probability vs Root Runs (Logistic Curve)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('joe_root_analysis_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional visualization: Performance metrics by venue
        fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        venue_stats_list = []
        for venue in self.data['Ground'].unique():
            venue_data = self.data[self.data['Ground'] == venue]
            
            innings = len(venue_data)
            total_runs = venue_data['Runs'].sum()
            
            if 'Not Out' in venue_data.columns:
                dismissals = len(venue_data[venue_data['Not Out'] == False])
            else:
                dismissals = innings
            
            avg = total_runs / dismissals if dismissals > 0 else total_runs
            
            venue_stats_list.append({
                'Ground': venue,
                'Innings': innings,
                'Avg Runs': avg,
                'High Score': venue_data['Runs'].max(),
                'Win Rate': venue_data['Win'].mean() * 100
            })
        
        venue_stats = pd.DataFrame(venue_stats_list).set_index('Ground')
        venue_stats = venue_stats.sort_values('Avg Runs', ascending=False)
        
        display_data = []
        for idx, row in venue_stats.iterrows():
            display_data.append([
                f"{int(row['Innings'])}",
                f"{int(row['Avg Runs'])}",
                f"{int(row['High Score'])}",
                f"{int(row['Win Rate'])}%"
            ])
        
        table = ax.table(cellText=display_data,
                        colLabels=['Innings', 'Average', 'High Score', 'Win Rate'],
                        rowLabels=venue_stats.index,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.3, 1.8)
        
        for i in range(len(['Innings', 'Average', 'High Score', 'Win Rate'])):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i, venue in enumerate(venue_stats.index, 1):
            if venue in ['Headingley', 'Edgbaston', "Lord's", 'Old Trafford', 'The Oval', 
                        'Trent Bridge', 'Southampton']:
                table[(i, -1)].set_facecolor('#e8f4f8')
            else:
                table[(i, -1)].set_facecolor('#f8e8e8')
        
        ax.axis('off')
        ax.set_title('Joe Root Performance Summary by Venue vs India (Home & Away)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.text(0.5, -0.05, 'Blue: Home (England) | Red: Away', 
                transform=ax.transAxes, ha='center', fontsize=10, style='italic')
        
        plt.savefig('joe_root_venue_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # NEW VISUALIZATION: 2025 Series Predictions
        self.create_2025_predictions_visual()
    
    def create_2025_predictions_visual(self):
        """Create visualization for 2025 series predictions"""
        venues_2025 = ['Headingley', 'Edgbaston', "Lord's", 'Old Trafford', 'The Oval']
        venue_full_names = {
            'Headingley': 'Headingley, Leeds',
            'Edgbaston': 'Edgbaston, Birmingham',
            "Lord's": "Lord's, London",
            'Old Trafford': 'Old Trafford, Manchester',
            'The Oval': 'The Oval, London'
        }
        
        predictions = []
        
        recent_form = self.data.tail(5)['Runs'].mean()
        
        for venue in venues_2025:
            if venue not in self.label_encoders['Ground'].classes_:
                venue_avg = self.data['Runs'].mean()
                print(f"Note: No historical data for {venue}, using overall average")
            else:
                venue_data = self.data[self.data['Ground'] == venue]
                venue_avg = venue_data['Runs'].mean() if len(venue_data) > 0 else self.data['Runs'].mean()
            
            for innings in [1, 2]:
                try:
                    predicted_runs = self.predict_root_runs(
                        venue=venue,
                        innings_number=innings,
                        form_last_5=recent_form,
                        year=2025
                    )
                    
                    win_prob = self.win_probability(predicted_runs, venue, innings)
                    
                    predictions.append({
                        'Venue': venue_full_names[venue],
                        'Venue_Short': venue,
                        'Innings': innings,
                        'Predicted_Runs': predicted_runs,
                        'Win_Probability': win_prob,
                        'Match_Result': 'WIN' if win_prob > 0.5 else 'LOSS'
                    })
                except:
                    predictions.append({
                        'Venue': venue_full_names[venue],
                        'Venue_Short': venue,
                        'Innings': innings,
                        'Predicted_Runs': venue_avg,
                        'Win_Probability': 0.5,
                        'Match_Result': 'DRAW'
                    })
        
        pred_df = pd.DataFrame(predictions)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        venues_order = ['Headingley, Leeds', 'Edgbaston, Birmingham', 
                       "Lord's, London", 'Old Trafford, Manchester', 'The Oval, London']
        
        x = np.arange(len(venues_order))
        width = 0.35
        
        innings1_data = pred_df[pred_df['Innings'] == 1].set_index('Venue').reindex(venues_order)
        innings2_data = pred_df[pred_df['Innings'] == 2].set_index('Venue').reindex(venues_order)
        
        bars1 = ax1.bar(x - width/2, innings1_data['Predicted_Runs'], width, 
                        label='1st Innings', color='#1f77b4', alpha=0.8)
        bars2 = ax1.bar(x + width/2, innings2_data['Predicted_Runs'], width, 
                        label='2nd Innings', color='#ff7f0e', alpha=0.8)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_xlabel('Venue', fontsize=12)
        ax1.set_ylabel('Predicted Runs', fontsize=12)
        ax1.set_title('Joe Root - Predicted Runs for 2025 Series vs India', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([v.split(',')[0] for v in venues_order], rotation=0)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')