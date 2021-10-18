import logging
import os.path
import re
from io import StringIO
from typing import List
import sys
import numpy as np
import pandas as pd
import scipy
import sklearn.neighbors._base
from plotnine import ggplot, geom_hline, geom_vline, geom_point, geom_text, aes, scale_color_manual, labs, theme, \
    ggsave, geom_line, scale_color_gradient, facet_wrap, geom_boxplot, scale_fill_manual, xlab, ylab, element_text, \
    geom_col, coord_flip, element_blank, scale_fill_gradient, geom_tile, scale_fill_gradientn
import matplotlib.pylab as plt
from scipy.stats import ttest_ind, f_oneway, fisher_exact
from sklearn.decomposition import PCA
import skfuzzy as fuzz
from statsmodels.stats.multitest import multipletests
import random

from uniprot.parser import UniprotSequence, UniprotParser

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from missingpy import MissForest

format_time = "%(asctime)s: %(message)s"


class Experiment:
    pattern_replicate = re.compile("(\d+)$")

    def __init__(self, name, condition=None, replicate=None):
        self.name = name
        self.condition = condition

        if replicate:
            self.replicate = replicate
        else:
            s = self.pattern_replicate.search(self.name)
            if s:
                self.replicate = s.group(1)
            else:
                self.replicate = "0"

    def __repr__(self):
        return f"Name: {self.name} - Condition: {self.condition}"

    def __str__(self):
        return f"{self.name} ({self.condition})"


class Analysis:
    experiments: list[Experiment]
    discrete_color_premade: list[str] = ["#998ec3", "#A3C0A6", "#f1a340", "#BA55D3",
                                         "#DA724A"]
    gradient_color_premade = {
        3: [
            (0, "#998ec3"),
            (0.5, "#f7f7f7"),
            (1, "#f1a340")
        ],
        2: [
            (0, "#f7f7f7"),
            (1, "#f1a340")
        ]
    }

    def __init__(self, conditions, gradient_colors=None, discrete_colors=None):
        self.conditions = conditions
        # if conditions is None:
        #     conditions = []
        # self.experiments = []
        # if log_file:
        #     assert isinstance(log_file, str)
        #     logging.basicConfig(filename=log_file, format=format_time, level=logging.DEBUG,
        #                         datefmt="%Y/%m/%d %I:%M:%S %p")
        # else:
        #     logging.basicConfig(filename="pyxis.log", format=format_time, level=logging.DEBUG,
        #                         datefmt="%Y/%m/%d %I:%M:%S %p")

        # self._experiment_condition_dict = {}
        # if len(conditions) != len(self.experiments):
        #     self.conditions = sorted(conditions, key=len, reverse=True)
        #     for e in self.experiments:
        #         for c in self.conditions:
        #             if c in e.name:
        #                 self._experiment_condition_dict[e.name] = c
        #                 break
        # else:
        #     self.conditions = conditions
        #     for e, c in zip(self.experiments, self.conditions):
        #         self._experiment_condition_dict[e.name] = c

        uqn_conditions = np.unique(self.conditions)
        self.experiments = []
        self.discrete_colors = {"Insignificant": "#AFAFAF"}
        if discrete_colors:
            self.assign_discrete_colors(discrete_colors, uqn_conditions)
        else:
            self.assign_discrete_colors(self.discrete_color_premade, uqn_conditions)

        if gradient_colors:
            self.gradient_colors = gradient_colors
        else:
            self.gradient_colors = self.gradient_color_premade
        self.data = None

    def assign_discrete_colors(self, discrete_colors, uqn_conditions):
        assert isinstance(discrete_colors, (tuple, list))
        if len(uqn_conditions) <= len(discrete_colors):
            for i in range(len(uqn_conditions)):
                self.discrete_colors[uqn_conditions[i]] = discrete_colors[i]
        elif len(uqn_conditions) > len(discrete_colors):
            last_post = 0
            for i in range(len(discrete_colors)):
                self.discrete_colors[uqn_conditions[i]] = discrete_colors[i]
                last_post = i
            for i in range(last_post, len(uqn_conditions)):
                self.discrete_colors[uqn_conditions[i]] = random_color()


class Data:
    def __init__(self, df=None, file_path=None, parent=None, operation=None, history=False, index=None):
        self.plot = None
        self.parent = parent
        self.file_path = file_path
        if self.file_path:
            self.sep = ","
            if self.file_path.endswith(".txt"):
                self.sep = "\t"
            if not index:
                self.df = pd.read_csv(self.file_path, sep=self.sep)
            else:
                self.df = pd.read_csv(self.file_path, sep=self.sep, index_col=[index])
        else:
            self.df = df
            if index:
                self.df = self.df.set_index([index])
        self.current_df_position = 0
        self.current_df = self.df.copy()

        if history:
            self.history = [Data(df=self.current_df)]
        else:
            self.history = []

        if not operation:
            self.operation = "Initiated"
        else:
            self.operation = operation

    def initiate_history(self):
        self.history = [self]

    def rewind(self, offset=1):
        offset = offset
        if len(self.history) == 0:
            raise ValueError(f"self.history must be inititated")
        position = self.current_df_position - offset
        if position >= 0:
            self.current_df_position = position
            self.current_df = self.history[position].current_df
            return self, self.current_df
        else:
            if self.parent is None:
                raise ValueError(
                    f"Offset {offset} is larger than possible rewind distance {len(self.history[:self.current_df_position])}")
            else:
                self.parent.current_df_position = len(self.parent.history) - 1
                self.parent.current_df = self.parent.history[-1].current_df
                rewind_leftover = offset - self.current_df_position

                return self.parent.rewind(rewind_leftover)

    def forward(self, offset=1):
        if len(self.history) == 0:
            raise ValueError(f"self.history must be inititated")
        if len(self.history) != 1:
            if self.current_df_position + offset < len(self.history):
                self.current_df_position = self.current_df_position + offset
                self.current_df = self.history[self.current_df_position].current_df
                return self, self.current_df
            else:
                forward_leftover = offset - len(self.history[-1].history) + 1
                self.history[-1].current_df_position = 0
                self.history[-1].current_df = self.history[-1].history[0].current_df
                return self.history[-1].forward(forward_leftover)
                # raise ValueError(
                #    f"Offset {offset} is larger than possible forward distance {len(self.history[self.current_df_position:])}")

        else:
            if offset > 0:
                raise ValueError(
                    f"Offset {offset} is larger than possible forward distance {len(self.history[self.current_df_position:])}")
            else:
                return self, self.current_df

    def get_columns(self, columns, branch=False):
        df = self.current_df[columns]
        a = Data(df=df, parent=self, operation=f"Get column(s) {','.join(columns)}")
        self._move_time(a, df)
        if branch:
            a.initiate_history()
            return a

    def _move_time(self, node, df):
        self.history.append(node)
        self.current_df_position = self.current_df_position + 1
        self.current_df = df.copy()

    def add_columns(self, column, origin_df=None, branch=False):
        df = self.current_df
        if not isinstance(origin_df, pd.DataFrame):
            for c in column:
                df[c.name] = c
        else:
            df = pd.concat([df, origin_df], axis=1)

        a = Data(df=df, parent=self, operation=f"Added column(s) {','.join([c.name for c in column])}")
        self._move_time(a, df)
        if branch:
            a.initiate_history()
            return a

    def get_all_operations(self, operation=[], start=True):

        if start:
            if self.current_df_position != 0:
                operation = []
                h = self.history[:self.current_df_position + 1]
                for o in h[::-1]:
                    operation.append(o.operation)
        if len(self.history) == 1:
            operation.append(self.operation)
        # else:
        #    operation = operation + [self.operation]
        if self.parent is not None:

            d = self.parent.history[:-1]
            for a in d[::-1]:
                operation.append(a.operation)
            self.parent.current_df_position = 0
            self.parent.current_df = self.parent.history[0].current_df
            return self.parent.history[0].get_all_operations(operation=operation, start=False)
        else:
            return operation

    def __repr__(self):
        return self.operation

    def remove_rows(self, conditions: dict, keep=False, branch=False):
        df = self.current_df
        operation = []
        for c in conditions:
            if c not in self.current_df.columns:
                raise ValueError(f"{c} is not a column in current data frame")
            else:
                operator = operator_dict[conditions[c]["operator"]]
                mask = operator(df[c], conditions[c]["value"])
                operation.append(f"{c}{conditions[c]['operator']}{conditions[c]['value']}")
                if keep:
                    df = df[mask]
                else:
                    df = df[~mask]

        if keep:
            operation = f"Removed row(s) that does not have {','.join(operation)}"
        else:
            operation = f"Removed row(s) that have {','.join(operation)}"
        a = Data(df=df, parent=self, operation=operation)
        self._move_time(a, df)
        if branch:
            a.initiate_history()
            return a

    def impute_lcm(self, experiments, shift, width, conditions, missing_operator, missing_criteria, branch=False):
        operator = operator_dict[missing_operator]
        df = self.current_df[experiments]

        median = df.median(axis=0)
        std = df.std(axis=0)
        data = {}
        for i in experiments:
            data[i] = np.random.normal(loc=median[i] - shift * std[i], scale=std[i] * width, size=len(df.index))

        df_new = pd.DataFrame(data, columns=experiments)
        df_fin = df.copy()
        self.replace_missing(conditions, df_fin, df_new, experiments, missing_criteria, operator)
        operation = f"Imputed missing data from conditions where number of missing data is {missing_operator} {missing_criteria} using left centered median"
        a = Data(df=df_fin, parent=self, operation=operation)
        self._move_time(a, df_fin)
        if branch:
            a.initiate_history()
            return a

    def replace_missing(self, conditions, df_fin, df_new, experiments, missing_criteria, operator):
        for i, r in df_fin.iterrows():
            d = count_missing(r, experiments, conditions)
            for i2 in range(len(experiments)):
                if pd.isnull(r[experiments[i2]]):
                    if operator(d[conditions[i2]], missing_criteria):
                        df_fin.at[i, experiments[i2]] = df_new.at[i, experiments[i2]]

    def impute_missing_forest(self, experiments, conditions, missing_operator, missing_criteria, branch=False):
        operator = operator_dict[missing_operator]
        df = self.current_df[experiments]
        imputer = MissForest()
        df_new = pd.DataFrame(imputer.fit_transform(df))
        df_fin = df.copy()
        self.replace_missing(conditions, df_fin, df_new, experiments, missing_criteria, operator)
        operation = f"Imputed missing data from conditions where number of missing data is {missing_operator} {missing_criteria} using MissForest"
        a = Data(df=df_fin, parent=self, operation=operation)
        self._move_time(a, df_fin)
        if branch:
            a.initiate_history()
            return a

    def impute_missing(self, experiments, conditions, min_per_condition, min_condition, branch=False):
        df = self.current_df
        cond = []
        for i, r in df.iterrows():
            a = count_missing(r, experiments, conditions)
            count_good_condition = 0

            for n in a:
                if a[n] > min_per_condition:
                    count_good_condition += 1

            if count_good_condition > min_condition:
                cond.append(True)
            else:
                cond.append(False)
        df["cond"] = pd.Series(cond, index=df.index)
        df = df[df["cond"] == True]
        operation = f"Removed rows with less than {min_condition} viable conditions"
        a = Data(df=df[experiments], parent=self, operation=operation)
        self._move_time(a, df)
        if branch:
            a.initiate_history()
            return a

    def print_procedure(self):
        a = self.get_all_operations()
        for i, n in enumerate(reversed(a), start=1):
            print(f"{i}/ {n}")

    def normalize(self, experiments, normalizer_mask=None, branch=False, method=None):
        assert method in ["median", "mean", "z-score", "z-score-col"]
        df = self.current_df[experiments]
        new_df = df.copy()

        if method in ["median", "mean"]:
            factor = pd.Series([])
            if method == "median":
                if isinstance(normalizer_mask, pd.DataFrame):
                    filtered = new_df[normalizer_mask]
                    factor = filtered.median(axis=0)
                else:
                    factor = new_df.median(axis=0)
            elif method == "mean":
                if isinstance(normalizer_mask, pd.DataFrame):
                    filtered = new_df[normalizer_mask]
                    factor = filtered.median(axis=0)
                else:
                    factor = new_df.mean(axis=0)
            for e in experiments:
                new_df[e] = new_df[e] - factor[e]
        elif method == "z-score":
            new_df = calculate_z_score(new_df.T).T
        elif method == "z-score-col":
            new_df = calculate_z_score(new_df)
        else:
            raise ValueError(f"{method} cannot be used with Pyxis")
        operation = f"Normalized dataset using {method}"
        a = Data(df=new_df, parent=self, operation=operation)
        self._move_time(a, new_df)
        if branch:
            a.initiate_history()
            return a

    def two_sample(self, comparisons, variance, conditions, experiments, branch=False):
        condition_dict = {}
        for c in comparisons:
            condition_dict[f"{c[0]}-{c[1]}"] = {c[0]: [], c[1]: []}

        for c, e in zip(conditions, experiments):
            for cd in condition_dict:
                if c in condition_dict[cd]:
                    condition_dict[cd][c].append(e)
        result = []
        for c in comparisons:
            exp = condition_dict[f"{c[0]}-{c[1]}"][c[0]] + condition_dict[f"{c[0]}-{c[1]}"][c[1]]
            df = self.current_df[exp]
            for i, r in df.iterrows():
                a = r[condition_dict[f"{c[0]}-{c[1]}"][c[0]]].dropna()
                b = r[condition_dict[f"{c[0]}-{c[1]}"][c[1]]].dropna()
                res = ttest_ind(a=a, b=b, equal_var=variance)
                df.at[i, "p-value"] = res[1]

            df["FC"] = df[condition_dict[f"{c[0]}-{c[1]}"][c[0]]].mean(axis=1) / df[condition_dict[f"{c[0]}-{c[1]}"][c[1]]].mean(axis=1)
            df["log2FC"] = np.log2(np.abs(df["FC"]))
            df["Comparison"] = f"{c[0]}-{c[1]}"
            result.append(df.reset_index())
        operation = f"Performed two-sided T-test"
        df = pd.concat(result, ignore_index=True)
        a = Data(df=df, parent=self, operation=operation)
        self._move_time(a, df)
        if branch:
            a.initiate_history()
            return a

    def anova(self, conditions, experiments, branch=False):
        condition_dict = {}
        for c, e in zip(conditions, experiments):
            if c not in condition_dict:
                condition_dict[c] = []
            condition_dict[c].append(e)

        df = self.current_df[experiments]
        for i, r in df.iterrows():
            samples = []
            for c in conditions:
                samples.append(r[condition_dict[c]].dropna())
            res = f_oneway(*samples)
            df.at[i, "p-value"] = res[1]

        operation = f"Performed one way ANOVA using data from {','.join(conditions)}"
        a = Data(df=df, parent=self, operation=operation)
        self._move_time(a, df)
        if branch:
            a.initiate_history()
            return a

    def p_correct(self, alpha, correction_method, branch=False):
        df = self.current_df
        if "Comparisons" in df.columns:
            result = []
            for i, g in df.groupby("Comparisons"):
                g = g[pd.notnull(g["p-value"])]
                a = p_correct(g["p-value"].values, alpha, correction_method)
                g["adj.p-value"] = pd.Series(a[1], index=g.index)

                for i2, r in g.iterrows():
                    g.at[i2, "adj.p-value"] = r["adj.p-value"]
                result.append(g)
            df = pd.concat(result, ignore_index=True)
        else:
            new_df = df.copy()
            new_df = new_df[pd.notnull(new_df["p-value"])]
            a = p_correct(new_df["p-value"].values, alpha, correction_method)
            new_df["adj.p-value"] = pd.Series(a[1], index=new_df.index)

            for i, r in new_df.iterrows():
                df.at[i, "adj.p-value"] = r["adj.p-value"]

        operation = f"Performed multipletesting using {correction_method}"
        a = Data(df=df, parent=self, operation=operation)
        self._move_time(a, df)
        if branch:
            a.initiate_history()
            return a

    def fuzzy_c(self, experiments=[], conditions=[], threshold=0, color_dict=None, branch=False, **kwargs):
        if not color_dict:
            color_dict = {}
        pca = PCA(n_components=2)
        a = self.current_df[experiments]
        a = calculate_z_score(a.T).T
        df = pca.fit_transform(a.values)
        df = pd.DataFrame(df, columns=["PC1", "PC2"])
        df.index = self.current_df.index

        sa = np.vstack((df["PC1"], df["PC2"]))

        list_functional_principal_components = []
        # fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
        # for number_of_centers, ax in enumerate(axes1.reshape(-1), 2):
        #     center, u, u0, d, jm, p, functional_principal_components = fuzz.cluster.cmeans(
        #         sa, number_of_centers, 2, error=0.005, maxiter=1000, init=None)
        #     list_functional_principal_components.append(functional_principal_components)
        #     cluster_membership = np.argmax(u, axis=0)
        #     for j in range(number_of_centers):
        #         ax.plot(df["PC1"][cluster_membership == j],
        #                 df["PC2"][cluster_membership == j],
        #                 '.')
        #     for pt in center:
        #         ax.plot(pt[0], pt[1], 'rs')
        #     ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(number_of_centers, functional_principal_components))
        #     ax.axis('off')
        # fig1.tight_layout()
        # fig2, ax2 = plt.subplots()
        # ax2.plot(np.r_[2:11], list_functional_principal_components)
        # ax2.set_xlabel("Number of centers")
        # ax2.set_ylabel("Fuzzy partition coefficient")
        # plt.show()
        current_fpc = (0, 0)
        for i in range(2, 10):
            center, u, u0, d, jm, p, functional_principal_components = fuzz.cluster.cmeans(
                sa, i, 2, error=0.005, maxiter=1000, init=None)
            if functional_principal_components > current_fpc[0]:
                current_fpc = (i, functional_principal_components)
                list_functional_principal_components.append((i, functional_principal_components))

            pass

        center, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
            sa, current_fpc[0], 2, error=0.005, maxiter=10000)
        df_fuzz = pd.DataFrame()
        for i in range(current_fpc[0]):
            df_fuzz_tmp = pd.DataFrame({"PC1": sa[0, u_orig.argmax(axis=0) == i],
                                        "PC2": sa[1, u_orig.argmax(axis=0) == i],
                                        "Cluster": str(i)})
            df_fuzz = pd.concat([df_fuzz, df_fuzz_tmp], axis=0)

        df_fuzz = df_fuzz.reset_index(drop=True)
        mask = pd.DataFrame(u_orig).T
        mask.index = df.index
        mask["member"] = mask.max(axis=1)
        df = df.sort_values(["PC1"])
        df = df.sort_values(["PC2"])
        df_fuzz = df_fuzz.sort_values(["PC1"])
        df_fuzz = df_fuzz.sort_values(["PC2"])
        df_fuzz.index = df.index
        df_fuzzy_cluster = pd.concat([self.current_df, df_fuzz, mask], axis=1)
        df_fuzzy_cluster.loc[(df_fuzzy_cluster["member"] < threshold), "Cluster"] = 'Low memb.'
        df_fuzzy_cluster.loc[(df_fuzzy_cluster["member"] < threshold), "member"] = 0
        df_fuzz["Cluster"] = df_fuzzy_cluster["Cluster"]

        condition_dict = {}
        for c, e in zip(conditions, experiments):
            if c not in condition_dict:
                condition_dict[c] = []
            condition_dict[c].append(e)

        for i, r in df_fuzzy_cluster.iterrows():
            if r["member"] < threshold:
                df_fuzzy_cluster.at[i, "Cluster"] = "Low memb."
                df_fuzzy_cluster.at[i, "member"] = 0
                if r["Cluster"] not in color_dict:
                    color_dict[r["Cluster"]] = random_color()
                df_fuzzy_cluster.at[i, "color"] = color_dict[r["Cluster"]]
            for k in condition_dict:
                df_fuzzy_cluster.at[i, k] = r[condition_dict[k]].mean()
        if "filename" in kwargs:
            df_fuzzy_cluster["id"] = df_fuzzy_cluster.index
            m = df_fuzzy_cluster.melt(id_vars=["Cluster", "member", "id"], value_vars=list(condition_dict.keys()))
            m = m.rename(columns={'variable': 'Condition', 'value': 'Z-Score'})
            m['Condition'] = pd.Categorical(m['Condition'], categories=list(condition_dict.keys()))
            number_squares = len(list(dict.fromkeys(m["Cluster"].tolist())))
            number_columns = 5
            if number_squares < number_columns:
                number_columns = current_fpc[0]
            else:
                number_columns = number_columns
            number_rows = np.ceil(current_fpc[0] / number_columns)
            if "size_box_width" in kwargs:
                size_box_width = kwargs["size_box_width"]
            else:
                size_box_width = 10
            if "size_box_height" in kwargs:
                size_box_height = kwargs["size_box_height"]
            else:
                size_box_height = 10

            size_width = number_columns * size_box_width
            size_height = number_rows * size_box_height
            # Set the colors for membership clusters
            if "gradient_colors" in kwargs:
                list_color_gradient = kwargs["gradient_colors"]
            else:
                list_color_gradient = ["#f7f7f7", "#f1a340"]
            # Draw Fuzzy clusters
            figure_fuzzy_clusters = (
                    ggplot(m, aes(x='Condition', y='Z-Score', color="member"))
                    + geom_point()
                    + geom_line(aes(group='id'))
                    + scale_color_gradient(low=list_color_gradient[0], high=list_color_gradient[1])
                    + labs(x='Condition', y='Z-Score', title='Fuzzy c means clustering', color='member')
                    + facet_wrap("Cluster", scales='free', ncol=number_columns)
                    + theme(legend_position="right",
                            legend_direction="vertical",
                            legend_title_align="center",
                            legend_box_spacing=0.4,
                            subplots_adjust={'hspace': 0.4, 'wspace': 0.25},
                            figure_size=(size_width, size_height), ))
            self.plot = figure_fuzzy_clusters
            ggsave(self.plot, kwargs["filename"])

        operation = f"Performed fuzzy C clustering"
        a = Data(df=df_fuzzy_cluster, parent=self, operation=operation)
        self._move_time(a, df_fuzzy_cluster)
        if branch:
            a.initiate_history()
            return a

    def volcano_plot(self, pvalue=0, logfc=0, text_column="", display_text=False, text_font_size=5, color_dict=None,
                     branch=False, **kwargs):
        assert "p-value" in self.current_df.columns and "log2FC" in self.current_df.columns
        log_p = 0
        if pvalue != 0:
            log_p = -np.log10(pvalue)

        df = self.current_df.copy()
        title = "Volcano Plot"
        if "title" in kwargs:
            title = kwargs["title"]
        pvalue_col = "p-value"
        if "adj.p-value" in df.columns:
            pvalue = "adj.p-value"
        if not color_dict:
            color_dict = {"Insignificant": "#AFAFAF", "Significant": "#998ec3"}

        df["-log10(p-value)"] = -np.log10(df[pvalue_col])
        if text_column == "":
            df["names"] = df.index
            text_column = "names"

        for i, r in df.iterrows():

            if r[pvalue_col] <= pvalue and abs(r["log2FC"]) >= logfc:
                df.at[i, "label"] = "Significant"
                df.at[i, "color"] = color_dict["Significant"]
            else:
                df.at[i, "label"] = "Insignificant"
                df.at[i, "color"] = color_dict["Insignificant"]
                df.at[i, "names"] = ""
        if display_text:

            fig_volcano_plot = (
                    ggplot(df, aes(**dict(x="log2FC", y="-log10(p-value)"))) +
                    geom_point(aes(**dict(color="label"))) +
                    scale_color_manual(df["color"].unique()) +
                    geom_text(
                        aes(**dict(label=text_column)),
                        size=text_font_size,
                        nudge_y=(df["-log10(p-value)"].max() / 50)) +
                    labs(**dict(x='log2FC',
                                y="-log10(p-value)",
                                color='Status',
                                title=title
                                )) +
                    geom_hline(yintercept=log_p, color='#c8c8c8') +
                    geom_vline(xintercept=-logfc, color='#c8c8c8') +
                    geom_vline(xintercept=logfc, color='#c8c8c8') +
                    theme(
                        legend_position="bottom",
                        legend_direction="horizontal",
                        legend_title_align="center",
                        legend_box_spacing=0.4,
                        figure_size=(8, 10))
            )
        else:
            fig_volcano_plot = (
                    ggplot(df, aes(**dict(x="log2FC", y="-log10(p-value)"))) +
                    geom_point(aes(**dict(color="label"))) +
                    scale_color_manual(df["color"].unique()) +
                    labs(**dict(x='log2FC',
                                y="-log10(p-value)",
                                color='Status',
                                title=title
                                )) +
                    geom_hline(yintercept=log_p, color='#c8c8c8') +
                    geom_vline(xintercept=-logfc, color='#c8c8c8') +
                    geom_vline(xintercept=logfc, color='#c8c8c8') +
                    theme(
                        legend_position="bottom",
                        legend_direction="horizontal",
                        legend_title_align="center",
                        legend_box_spacing=0.4,
                        figure_size=(8, 10))
            )

        self.plot = fig_volcano_plot
        if "filename" in kwargs:
            ggsave(plot=self.plot, filename=kwargs["filename"], dpi=150, limitsize=False, verbose=False)

        operation = f"Plot volcano plot with p-value cutoff {pvalue} and logFC cutoff {logfc}"
        a = Data(df=df, parent=self, operation=operation)
        self._move_time(a, df)
        if branch:
            a.initiate_history()
            return a

    def box_plot(self, experiments=[], conditions=[], by_label=None, label_col=None, discrete_colors=None,
                 group_col=None, branch=False, **kwargs):
        condition_dict = {}
        reverse_dict = {}

        for c, e in zip(conditions, experiments):
            if c not in condition_dict:
                condition_dict[c] = []
            condition_dict[c].append(e)
            reverse_dict[e] = c

        id_vars = []
        save_cols = []
        if label_col:
            save_cols.append(label_col)
            id_vars.append(label_col)
        if group_col:
            save_cols.append(group_col)
            id_vars.append(group_col)
        df = self.current_df[experiments + save_cols]
        id_col = ""
        if df.index.name != "":
            id_vars.append(df.index.name)
            id_col = df.index.name
            df.reset_index(inplace=True)

        if id_col:
            melted = df.melt(id_vars=id_vars, value_vars=experiments, var_name="Experiment", value_name="Intensity")

        else:
            if group_col:
                melted = df.melt(id_vars=group_col, var_name="Experiment", value_vars=experiments,
                                 value_name="Intensity")
            else:
                melted = df.melt(var_name="Experiment", value_vars=experiments,
                                 value_name="Intensity")

        melted["Condition"] = pd.Series([reverse_dict[i] for i in melted["Experiment"]], index=melted.index)
        melted["Condition"] = pd.Categorical(melted["Condition"],
                                             categories=np.unique(conditions))
        melted["Experiment"] = pd.Categorical(melted["Experiment"],
                                              categories=experiments)
        if "filename" in kwargs:
            root, fi = os.path.split(kwargs["filename"])
            filename, extension = fi.split(".")
            if group_col:
                for i, g in melted.groupby(group_col):
                    self._create_boxplot(by_label, discrete_colors, label_col, g)
                    ggsave(self.plot, kwargs["filename"].replace(extension, f"group_{i}.{extension}"), limitsize=False)
            else:
                self._create_boxplot(by_label, discrete_colors, label_col, melted)
                ggsave(self.plot, kwargs["filename"], limitsize=False)
        if group_col:
            operation = f"Boxplot created with graphs seperated by {group_col}"
        else:
            operation = f"Boxplot created"
        a = Data(df=melted, parent=self, operation=operation)
        self._move_time(a, melted)
        if branch:
            a.initiate_history()
            return a

    def _create_boxplot(self, by_label, discrete_colors, label_col, melted):
        if discrete_colors:
            colors = []
            for i in melted["Condition"]:
                if i in discrete_colors:
                    colors.append(discrete_colors[i])
                else:
                    discrete_colors[i] = random_color()
                    colors.append(discrete_colors[i])
            melted["color"] = pd.Series(colors, index=melted.index)
            unique_colors = discrete_colors
            if by_label:
                size_box_width = 3
                size_box_height = 2

                number_squares = len(np.unique(melted[label_col]))
                number_columns = 10
                if number_squares < number_columns:
                    number_columns = number_squares
                else:
                    number_columns = number_columns
                number_rows = np.ceil(number_squares / number_columns)
                size_width = number_columns * size_box_width
                size_height = number_rows * size_box_height
                box_plot = (
                        ggplot(melted, aes(x="Condition", y="Intensity")) +
                        geom_boxplot(alpha=0.5, outlier_shape='') +
                        aes(color='Condition', fill='Condition') +
                        scale_color_manual(unique_colors) +
                        scale_fill_manual(unique_colors) +
                        xlab('') + ylab('log2 Intensities') +
                        facet_wrap(label_col, scales='free', ncol=number_columns) +
                        theme(legend_position="top",
                              legend_direction="horizontal",
                              legend_title_align="center",
                              legend_box_spacing=0.4,
                              subplots_adjust={'hspace': 0.4, 'wspace': 0.25},
                              figure_size=(size_width, size_height),
                              axis_text_x=element_text(angle=90, hjust=0.25)))

            else:
                box_plot = (
                        ggplot(melted, aes(x="Condition", y="Intensity")) +
                        geom_boxplot(alpha=0.5, outlier_shape='') +
                        aes(color='Condition', fill='Condition') +
                        # scale_color_manual(unique_colors) +
                        # scale_fill_manual(unique_colors) +
                        xlab('') + ylab('log2 Intensities') +
                        theme(
                            legend_position="top",
                            legend_direction="horizontal",
                            legend_title_align="center",
                            legend_box_spacing=0.4,
                            subplots_adjust={'hspace': 0.4, 'wspace': 0.25},
                            axis_text_x=element_text(angle=90, hjust=1)
                        )
                    # facet_wrap(displayed_name, scales='free', ncol=number_columns)
                )
        else:
            if by_label:
                size_box_width = 3
                size_box_height = 2

                number_squares = len(np.unique(melted[label_col]))
                number_columns = 5
                if number_squares < number_columns:
                    number_columns = number_squares
                else:
                    number_columns = number_columns
                number_rows = np.ceil(number_squares / number_columns)
                size_width = number_columns * size_box_width
                size_height = number_rows * size_box_height
                box_plot = (
                        ggplot(melted, aes(x="Condition", y="Intensity")) +
                        geom_boxplot(alpha=0.5, outlier_shape='') +
                        aes(color='Condition', fill='Condition') +
                        xlab('') + ylab('log2 Intensities') +
                        facet_wrap(label_col, scales='free', ncol=number_columns) +
                        theme(legend_position="top",
                              legend_direction="horizontal",
                              legend_title_align="center",
                              legend_box_spacing=1,
                              subplots_adjust={'hspace': 0.4, 'wspace': 0.25},
                              figure_size=(size_width, size_height),
                              axis_text_x=element_text(angle=90, hjust=1)))

            else:
                box_plot = (
                        ggplot(melted, aes(x="Condition", y="Intensity")) +
                        geom_boxplot(alpha=0.5, outlier_shape='') +
                        aes(color='Condition', fill='Condition') +
                        xlab('') + ylab('log2 Intensities') +
                        theme(
                            legend_position="top",
                            legend_direction="horizontal",
                            legend_title_align="center",
                            legend_box_spacing=0.4,
                            subplots_adjust={'hspace': 0.4, 'wspace': 0.25},

                            axis_text_x=element_text(angle=90, hjust=1)
                        )
                )
        self.plot = box_plot

    def rank_plot(self, experiments, conditions, condition_1, condition_2, pvalue=0.05, logfc=0.5, text_column="",
                  display_text=False, text_font_size=5, branch=False, discrete_colors=None, **kwargs):
        condition_dict = {}
        reverse_dict = {}
        pvalue_col = "p-value"
        if "adj.p-value" in self.current_df.columns:
            pvalue_col = "adj.p-value"
        for c, e in zip(conditions, experiments):
            if c not in condition_dict:
                condition_dict[c] = []
            condition_dict[c].append(e)
            reverse_dict[e] = c

        df = self.current_df.copy()

        for i, r in self.current_df.iterrows():
            df.at[i, condition_1] = r[condition_dict[condition_1]].mean()
            df.at[i, condition_2] = r[condition_dict[condition_2]].mean()
            if r[pvalue_col] <= pvalue:
                if r["log2FC"] < 0 and abs(r["log2FC"]) > logfc:
                    df.at[i, "Status"] = "Significantly Down-regulated"
                if r["log2FC"] > 0 and abs(r["log2FC"]) > logfc:
                    df.at[i, "Status"] = "Significantly Up-regulated"
            else:
                df.at[i, "Status"] = "Insignificant"
        df["Status"] = df["Status"].fillna("No comparisons data")
        if "filename" in kwargs:
            root, fi = os.path.split(kwargs["filename"])
            filename, extension = fi.split(".")
            for i in np.unique(df["Status"]):
                if i not in discrete_colors:
                    discrete_colors[i] = random_color()
            for i in [condition_1, condition_2]:
                df["Rank " + i] = df[i].rank()
                df["Rank " + i] = -(df["Rank " + i] - df["Rank " + i].max()) + 1
                insignificant = df[df["Status"] == "Insignificant"]
                significant = df[df["Status"] != "Insignificant"]
                N = 10

                if len(significant.index) > N * 2:
                    significant_top = significant["Rank " + i].nlargest(n=N, keep='first')
                    significant_bottom = significant["Rank " + i].nsmallest(n=N, keep='first')
                    significant_top_bottom = pd.concat([significant_top, significant_bottom])
                    significant_top_bottom = significant.loc[significant_top_bottom.index]
                else:
                    significant_top_bottom = significant
                if display_text:
                    self.plot = (ggplot() +
                                 geom_point(insignificant,
                                            aes(
                                                x='Rank ' + i,
                                                y=i,
                                                color="Status"
                                            ),
                                            size=1) +
                                 geom_point(
                                     significant,
                                     aes(
                                         x='Rank ' + i,
                                         y=i,
                                         color="Status"
                                     ),
                                     size=1) +
                                 scale_color_manual(discrete_colors) +
                                 geom_text(significant_top_bottom,
                                           aes(x='Rank ' + i,
                                               y=i,
                                               label=significant_top_bottom[text_column] + ' (' + (
                                                   significant_top_bottom['Rank ' + i].astype(int).astype(str)) + ')'),
                                           size=text_font_size,
                                           nudge_y=(df[i].max() / 50),
                                           nudge_x=(df['Rank ' + i].max() / 50),
                                           adjust_text={'expand_points': (2, 2), 'arrowprops': {'arrowstyle': '-'}}) +
                                 labs(x='Rank',
                                      y='log2 Intensity ' + i) +
                                 theme(figure_size=(4, 6))
                                 )
                else:
                    self.plot = (ggplot() +
                                 geom_point(insignificant,
                                            aes(
                                                x='Rank ' + i,
                                                y=i,
                                                color="Status"
                                            ),
                                            size=1) +
                                 geom_point(
                                     significant,
                                     aes(
                                         x='Rank ' + i,
                                         y=i,
                                         color="Status"
                                     ),
                                     size=1) +
                                 scale_color_manual(discrete_colors) +
                                 labs(x='Rank',
                                      y='log2 Intensity ' + i) +
                                 theme(figure_size=(4, 6))
                                 )
                ggsave(self.plot, kwargs["filename"].replace(extension, f"group_{i}.{extension}"))

        operation = f"Created rank plot with p-value cutoff {pvalue} and log2FC cutoff {logfc}"
        a = Data(df=df, parent=self, operation=operation)
        self._move_time(a, df)
        if branch:
            a.initiate_history()
            return a

    def go_terms_enrichment(self, target, acc_col_target, acc_col_source, methods,
                            gradients=["#998ec3", "#f7f7f7", "#f1a340"], alpha_value=0.05, branch=False, **kwargs):
        # Describe the different arguments

        df = self.current_df
        root_go_term = {
            "GO:008150": "BP", "GO:0003674": "MF", "GO:0005575": "CC"
        }
        go_stuff = {
            "GO:008150": set(),
            "GO:0003674": set(),
            "GO:0005575": set()
        }

        res = self.get_uniprot(acc_col_source, df)
        res_target = self.get_uniprot(acc_col_target, target)

        res["GO List"] = res["Gene ontology IDs"].str.split("; ")
        res_target["GO List"] = res_target["Gene ontology IDs"].str.split("; ")
        go_dict = {}
        go_cols = ["Gene ontology (biological process)", "Gene ontology (cellular component)",
                   "Gene ontology (molecular function)"]
        go_dict_target = {}

        self.count_go(go_cols, go_dict, go_stuff, res)
        self.count_go(go_cols, go_dict_target, go_stuff, res_target)
        go_stuff_total = {
            "GO:008150": {"background": 0, "target": 0},
            "GO:0003674": {"background": 0, "target": 0},
            "GO:0005575": {"background": 0, "target": 0}
        }
        result = []
        for g in go_stuff:
            go_stuff_total[g]["background"] = sum([i["count"] for i in go_dict.values()])
            go_stuff_total[g]["target"] = sum([i["count"] for i in go_dict_target.values()])
            for i in go_stuff[g]:
                if i in go_dict_target:
                    a = go_dict_target[i]["count"]
                    b = go_stuff_total[g]["target"] - a
                    c = go_dict[i]["count"] - go_dict_target[i]["count"]
                    d = go_stuff_total[g]["background"] - go_dict[i]["count"] - b
                    p = fisher_exact([[a, b], [c, d]])
                    result.append([i, root_go_term[g], go_stuff_total[g]["target"], go_stuff_total[g]["background"],
                                   ";".join(go_dict_target[i]["acc"]), p[1]])

        result = pd.DataFrame(result, columns=["GO IDs", "Pathway", "Background", "Target", "Proteins", "p-value"])
        result_corrected = []
        for i, g in result.groupby("Pathway"):
            for m in methods:
                p = p_correct(g["p-value"].values, method=m, alpha=alpha_value)
                g[m] = pd.Series(p[1], index=g.index)

            result_corrected.append(g)
        result_corrected = pd.concat(result_corrected, ignore_index=True)
        if "filename" in kwargs:
            pcutoff = -np.log10(alpha_value)
            root, fi = os.path.split(kwargs["filename"])
            filename, extension = fi.split(".")
            for i, g in result_corrected.groupby("Pathway"):
                for m in methods:
                    g = g[g[m] <= alpha_value]
                    if not g.empty:
                        g[f"-log10({m})"] = -np.log10(g[m])

                        figure_go_enrich = (ggplot() +
                                            geom_col(g, aes(x="GO IDs", y=f"-log10({m})", fill=f"-log10({m})"),
                                                     alpha=0.5) +
                                            scale_fill_gradient(colors=gradients,
                                                                limits=np.array([4 / 5 * pcutoff,
                                                                                 6 / 5 * pcutoff]), ) +

                                            geom_col(g, aes(x="GO IDs", y=f"-log10({m})"),
                                                     color='black',
                                                     alpha=0) +

                                            geom_hline(yintercept=pcutoff,
                                                       color='black',
                                                       size=1,
                                                       linetype='dashed') +
                                            coord_flip() + theme(figure_size=(3, len(g) * 0.175),
                                                                 legend_title=element_blank(), ))
                        ggsave(figure_go_enrich, kwargs["filename"].replace(extension, f"{i}{m}.{extension}"))

        operation = f"Performed GO term enrichment analysis with p-value cutoff {alpha_value}"
        a = Data(df=result_corrected, parent=self, operation=operation)
        self._move_time(a, result_corrected)
        if branch:
            a.initiate_history()
            return a

    def get_uniprot(self, acc_col, df):
        for i, r in df.iterrows():
            seq = UniprotSequence(r[acc_col], True)
            if seq.accession:
                df.at[i, acc_col] = str(seq)
        accessions = df[acc_col].unique()
        parser = UniprotParser(accessions, True)
        res = pd.DataFrame()
        for i in parser.parse("tab", method="post"):
            res = pd.read_csv(StringIO(i), sep="\t")
            res = res.rename(columns={res.columns[-1]: "query"})
        return res

    def count_go(self, go_cols, go_dict, go_stuff, res):
        for _, r in res.iterrows():
            for g in r["GO List"]:
                if g:
                    for i in range(3):
                        if pd.notnull(r[go_cols[i]]):
                            if g in r[go_cols[i]]:
                                if i == 0:
                                    go_stuff["GO:008150"].add(g)
                                elif i == 1:
                                    go_stuff["GO:0005575"].add(g)
                                else:
                                    go_stuff["GO:0003674"].add(g)
                                break
                    if g not in go_dict:
                        go_dict[g] = {"acc": set(), "count": 0}
                    go_dict[g]["count"] += 1
                    go_dict[g]["acc"].add(r["query"])

    def correlation_heatmap(self, experiments, conditions, gradients=["#998ec3", "#f7f7f7", "#f1a340"], branch=False, **kwargs):
        condition_dict = {}
        df = self.current_df[experiments]
        for c, e in zip(conditions, experiments):
            if c not in condition_dict:
                condition_dict[c] = []
            condition_dict[c].append(e)

        matrix = df.corr()
        if "filename" in kwargs:
            matrix.reset_index(inplace=True)
            # matrix = pd.DataFrame(matrix, columns=experiments)
            # matrix["y"] = pd.Series(experiments, index=matrix.index)
            m = matrix.melt(id_vars="index", value_vars=experiments, var_name="x", value_name="coefficient")
            m = m.rename(columns={"index": "y"})
            m["x"] = pd.Categorical(m["x"], experiments)
            m["y"] = pd.Categorical(m["y"], experiments)
            m["coeff_round"] = np.round(m["coefficient"], 2)
            se = (ggplot(m, aes(x="x", y="y", fill="coefficient")) +
                  geom_tile(aes(width=3, height=3)) +
                  geom_text(aes(label="coeff_round"), size=10, color="white") +
                  scale_fill_gradientn(colors=gradients,
                                      limits=np.array([m["coefficient"].min(), m["coefficient"].max()])) +
                  labs(x='', y='', title='Pearson coefficients') +
                  theme(legend_position="top",
                        legend_direction="horizontal",
                        legend_title_align="center",
                        legend_box_spacing=0.4,
                        subplots_adjust={'hspace': 0.4, 'wspace': 0.25},
                        axis_text_x=element_text(angle=90, hjust=0.25))
                  )
            ggsave(se, kwargs["filename"])
        operation = f"Created correlation matrix for heatmap"
        a = Data(df=matrix, parent=self, operation=operation)
        self._move_time(a, matrix)
        if branch:
            a.initiate_history()
            return a

    def correlation_scatter(self, experiments, conditions):
        condition_dict = {}
        df = self.current_df[experiments]
        for c, e in zip(conditions, experiments):
            if c not in condition_dict:
                condition_dict[c] = []
            condition_dict[c].append(e)

        for i in experiments:
            for i2 in experiments:
                if i != i2:
                    b = df[[i, i2]]
                    b = b.rename(columns={i: "x", i2: "y"})
                    a = np.corrcoef(b["x"].values, b["y"].values)
                    print(a[0, 1])


def p_correct(values, alpha, method):
    a = multipletests(values, alpha=alpha, method=method)
    return a


def calculate_z_score(df):
    for i in df.columns:
        df[i] = (df[i] - df[i].mean()) / df[i].std(ddof=1)

    return df


def count_missing(row, experiments, conditions):
    data = dict()
    for i in range(len(experiments)):
        if pd.notnull(row[experiments[i]]):
            if conditions[i] not in data:
                data[conditions[i]] = 0
            data[conditions[i]] += 1
    return data


def equal(source_df, target):
    return source_df == target


def lesser(source_df, target):
    return source_df < target


def greater(source_df, target):
    return source_df > target


def lesser_and_equal(source_df, target):
    return source_df <= target


def greater_and_equal(source_df, target):
    return source_df >= target


def random_color():
    color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    return color


operator_dict = {
    "==": equal,
    "<": lesser,
    ">": greater,
    ">=": greater_and_equal,
    "<=": lesser_and_equal
}
