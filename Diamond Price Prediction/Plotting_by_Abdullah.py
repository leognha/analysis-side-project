# copy from https://www.kaggle.com/code/abdmental01/plotting-masterclass 
# Basic Modules 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# Complete Plotting Class
class Plotting_by_Abdullah:
    def __init__(self, DATA):
        self.DATA = DATA 
   # Scatter plot With Hue and Not Hue 
    def scatter_plot_all(self, T_V, hue=None, palette=None):
        # Num Cols 
        num_cols = [col for col in self.DATA.columns if self.DATA[col].dtype != 'object' and self.DATA[col].dtype != 'category' and self.DATA[col].dtype != 'bool']

        # Features Except T_V
        FEATURE= [col for col in num_cols if col != T_V]
        num_features = len(FEATURE)

        # Calculate number of rows and columns for subplots
        ncols = min(3, num_features) 
        nrows = (num_features + ncols - 1) // ncols

        # Create subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))

        # Assign colors from the custom palette
        if palette is None:
            colors = sns.color_palette() 
        else:
            colors = palette

        # Plot each variable against the T_V
        for i, (FE, color) in enumerate(zip(FEATURE, colors)):
            if nrows == 1:
                ax = axes[i]
            else:
                ax = axes.flatten()[i]

            if hue:
                sns.scatterplot(x=self.DATA[FE], y=self.DATA[T_V], hue=self.DATA[hue], ax=ax, palette=palette)
                ax.set_title(f"{FE} vs {T_V} (Hue: {hue})")
                ax.set_xlabel(FE)
                ax.set_ylabel(T_V)
            else:
                sns.scatterplot(x=self.DATA[FE], y=self.DATA[T_V], ax=ax, palette=palette)
                ax.set_title(f"{FE} vs {T_V}")
                ax.set_xlabel(FE)
                ax.set_ylabel(T_V)

        # Hide empty subplots
        for i in range(num_features, nrows * ncols):
            if nrows == 1:
                fig.delaxes(axes[i])
            else:
                fig.delaxes(axes.flatten()[i])

        plt.tight_layout()
        plt.show()
        
    # Quick Overview of Plotting
    def QUICK_SHOW(self, color_palette=None, H=True, B=True, C=True, P=True):
        
        # Set the color palette if provided, otherwise use default
        if color_palette:
            sns.set_palette(color_palette)

        # Data Types 
        num_cols = self.DATA.select_dtypes(include=['float', 'integer']).columns.tolist()
        cat_cols = self.DATA.select_dtypes(include=['object', 'category']).columns.tolist()

        # Error handling for data types
        if not all(col in self.DATA.columns for col in num_cols + cat_cols):
            raise ValueError("Columns provided must exist in the dataframe.")

        # Error handling for empty data
        if self.DATA.empty:
            raise ValueError("Dataframe is empty.")

        num_rows = len(num_cols)
        cat_rows = len(cat_cols)
        total_rows = max(num_rows, cat_rows)

        if total_rows == 0:
            raise ValueError("No columns provided for plotting.")

        # Subplots Configure
        fig, axes = plt.subplots(total_rows, 4, figsize=(6*4, 6*total_rows))
        plt.subplots_adjust(hspace=0.5)

        # Hist and Box Plots 
        for i, col in enumerate(num_cols):
            if i < num_rows:
                if H:
                    sns.histplot(self.DATA[col], ax=axes[i, 0], kde=True)
                    axes[i, 0].set_title(f'Histogram of {col}')
                    axes[i, 0].set_xlabel(col)
                    axes[i, 0].set_ylabel('Frequency')
                if not H:
                    axes[i, 0].set_visible(False)
                if B:
                    sns.boxplot(self.DATA[col], ax=axes[i, 1])
                    axes[i, 1].set_title(f'Box Plot of {col}')
                    axes[i, 1].set_xlabel(col)
                    axes[i, 1].set_ylabel('')
                else:
                    axes[i, 1].set_visible(False)
            else:
                for j in range(2):
                    axes[i, j].set_visible(False)
        # Pie and Count Plot 
        for i, col in enumerate(cat_cols, start=0):
            # Convert the categorical column to pandas categorical data
            self.DATA[col] = pd.Categorical(self.DATA[col])

            counts = self.DATA[col].value_counts()

            # Plot Bar Chart
            if C:
                sns.countplot(data=self.DATA, x=col, ax=axes[i, 2], order=self.DATA[col].value_counts().index)
                axes[i, 2].set_title(f'Count Plot of {col}')
                axes[i, 2].set_xlabel(col)
                axes[i, 2].set_ylabel('Count')
                axes[i, 2].tick_params(axis='x', rotation=45)
            if not C:
                axes[i, 2].set_visible(False)

            # Plot PIE Chart
            if P:
                axes[i, 3].pie(counts, labels=counts.index, autopct='%1.1f%%')
                axes[i, 3].set_title(f'Pie Plot of {col}')
                axes[i, 3].set_xlabel('')
                axes[i, 3].set_ylabel('')

            # Hide axes for PIE Plot if disabled
            if not P:
                axes[i, 3].set_visible(False)

        # Clear remaining empty subplots for categorical columns
        for j in range(cat_rows, total_rows):
            for k in range(2, 4):
                axes[j, k].set_visible(False)

        plt.tight_layout()
        plt.show()
        
    # Volion Plots   
    def violin_plot(self, target_variable, palette=None):
        # Select numerical columns
        numerical_columns = [col for col in self.DATA.columns if self.DATA[col].dtype != 'object' and self.DATA[col].dtype != 'category' and self.DATA[col].dtype != 'bool']

        # Determine the number of features
        num_features = len(numerical_columns)

        # Determine the number of rows and columns for subplots
        ncols = min(3, num_features)
        nrows = (num_features + ncols - 1) // ncols

        # Create subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
        plt.subplots_adjust(hspace=0.5)

        # Plot violin plots for each numerical column
        for i, column in enumerate(numerical_columns):
            if nrows == 1:
                ax = axes[i]
            else:
                ax = axes.flatten()[i]
            sns.violinplot(x=target_variable, y=column, data=self.DATA, ax=ax, palette=palette)
            ax.set_title(f"Violin plot of {column} by {target_variable}")
            ax.set_xlabel(target_variable)
            ax.set_ylabel(column)

        # Hide empty subplots
        for i in range(num_features, nrows * ncols):
            if nrows == 1:
                fig.delaxes(axes[i])
            else:
                fig.delaxes(axes.flatten()[i]) 

        plt.tight_layout()
        plt.show()

    def violin_plot_Category(self, target_variable, palette=None):
        # Select categorical columns
        categorical_columns = [col for col in self.DATA.columns if self.DATA[col].dtype == 'object' or self.DATA[col].dtype == 'category']

        # Determine the number of features
        num_features = len(categorical_columns)

        # Determine the number of rows and columns for subplots
        ncols = min(3, num_features)
        nrows = (num_features + ncols - 1) // ncols

        # Create subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
        plt.subplots_adjust(hspace=0.5)

        # Flatten axes
        axes = np.ravel(axes)

        # Plot violin plots for each categorical column
        for i, column in enumerate(categorical_columns):
            ax = axes[i]

            # Plot only if the axis exists
            if ax is not None:
                sns.violinplot(x=column, y=target_variable, data=self.DATA, ax=ax, palette=palette)
                ax.set_title(f"Violin plot of {target_variable} by {column}")
                ax.set_xlabel(column)
                ax.set_ylabel(target_variable)

        # Hide empty subplots
        for ax in axes[num_features:]:
            # Remove the axis if it exists
            if ax is not None:
                fig.delaxes(ax)

        plt.tight_layout()
        plt.show()