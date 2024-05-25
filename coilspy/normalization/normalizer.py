import pandas as pd
import numpy as np

class CoilNormalizer():
    def __init__(self):
        super(CoilNormalizer, self).__init__()
        self.max_change_df = None
        self.conserved_subgroups = None
    
    def max_absolute_change(self, df_diff):
        # Calculate the absolute change for each feature
        abs_change = df_diff.abs()

        # Find the maximum absolute change for each feature
        max_changes = abs_change.max()

        # Convert the series to a DataFrame
        max_change_df = max_changes.to_frame(name='Max Absolute Change')

        return max_change_df
    
    def normalize(self, df):
        # Calculate the change for each feature
        df_diff = df.diff()

        # Drop first row
        df_diff = df_diff.iloc[1:,:]
        
        # Calculate the max absolute change for each feature
        max_change_df = self.max_absolute_change(df_diff)

        normalized_features = []
        conserved_subgroups = {}
        count = 0
        index = 0
        for column in df_diff.columns:
            max_change_val = max_change_df.loc[column, 'Max Absolute Change']
            normalized_feature = (df_diff[column] + max_change_val) / (2 * max_change_val)
            
            normalized_counter = 1 - normalized_feature
            
            normalized_features.append(normalized_counter)
            normalized_features.append(normalized_feature)
            
            conserved_subgroups[index] = [count, count + 1]
            index += 1
            count += 2
            
        # Create dataframe
        normalized_df = pd.DataFrame(normalized_features).T
        print(normalized_df)
                
        # Normalize the dataframe again by row such that each row sums to 1
        normalized_df = normalized_df.div(normalized_df.sum(axis=1), axis=0)
        
        self.max_change_df = max_change_df
        self.conserved_subgroups = conserved_subgroups
        return normalized_df
    
    def denormalize(self, normalized_df, initial_value):
        # For now assume there are even pairs
        denorm_dict = {}
        for key, value in self.conserved_subgroups.items():
            df_sel = normalized_df.iloc[:,value]
            denorm_dict[df_sel.columns[1]] = df_sel.div(df_sel.sum(axis=1), axis=0).iloc[:,1]
            
        denormalized_df = pd.DataFrame(denorm_dict, index=normalized_df.index)
        
        reconstructed_array = []
        new_value = initial_value
        reconstructed_array.append(initial_value)
        delta_max = self.max_change_df.iloc[:,0]
        for i in range(denormalized_df.shape[0]):
            delta_t = 2 * denormalized_df.iloc[i,:] * delta_max - delta_max
            new_value = new_value + delta_t
            reconstructed_array.append(new_value)
        return pd.DataFrame(reconstructed_array, index = [initial_value.name] + list(denormalized_df.index))
    
def segment_time_series(series, length):
    # Assuming series is a numpy array of shape [total_length, n_features]
    total_length, n_features = series.shape
    segments = []
    for start in range(0, total_length - length, length):
        segment = series[start:start + length]
        segments.append(segment)
    return np.stack(segments)