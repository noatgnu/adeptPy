from unittest import TestCase
from pyxis.Pyxis import Data, Experiment, Analysis
import pandas as pd
import numpy as np
import re


class TestData(TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.da = Data(file_path="proteinGroups.txt", history=True, index="Majority protein IDs")
        self.analysis = Analysis(["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "D", "D", "D", "E", "E"])

        self.experiments = []
        experiment_dict = {}
        for c in self.da.current_df.columns:
            if c.startswith("Intensity "):
                e = Experiment(c)
                experiment_dict[int(e.replicate)] = e
        a = list(experiment_dict.keys())
        a = sorted(a)
        for c in a:
            self.experiments.append(experiment_dict[c])
        self.conditions = ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "D", "D", "D", "E", "E"]

    def test_get_columns(self):
        columns = ["Protein IDs", "Peptide counts (all)", "Peptide counts (razor+unique)",
                   "Peptide counts (unique)", "Protein names", "Gene names", "Fasta headers", "Peptides",
                   "Razor + unique peptides", "Unique peptides", "Mol. weight [kDa]", "Sequence length",
                   "Sequence lengths", "Q-value", "Score", "Intensity", "MS/MS count", "Only identified by site",
                   "Reverse", "Potential contaminant", "id", "Peptide IDs", "Peptide is razor", "Mod. peptide IDs",
                   "Evidence IDs", "MS/MS IDs", "Best MS/MS"]
        self.da.get_columns(columns)
        assert len(columns) == len(self.da.current_df.columns)

    def test_add_columns(self):
        self.test_get_columns()
        self.da.add_columns([self.da.df[i] for i in self.da.df.columns if i.startswith("Intensity ")])

    def test_get_all_operations(self):
        columns = ["Protein IDs", "Peptide counts (all)", "Peptide counts (razor+unique)",
                   "Peptide counts (unique)", "Protein names", "Gene names", "Fasta headers", "Peptides",
                   "Razor + unique peptides", "Unique peptides", "Mol. weight [kDa]", "Sequence length",
                   "Sequence lengths", "Q-value", "Score", "Intensity", "MS/MS count", "Only identified by site",
                   "Reverse", "Potential contaminant", "id", "Peptide IDs", "Peptide is razor", "Mod. peptide IDs",
                   "Evidence IDs", "MS/MS IDs", "Best MS/MS"]
        a = self.da.get_columns(columns, branch=True)
        d = self.da.df[[e.name for e in self.experiments]]
        b = a.add_columns([d[e] for e in d.columns], branch=True)

        # print(b.history)
        c = b.get_all_operations()
        print(c)
        # assert c[-1] == "Initiated"
        # assert c[0].startswith("Added column(s) Intensity")

    def test_remove_rows(self):
        self.test_get_columns()
        self.test_add_columns()
        operation_remove = {
            "Reverse": {
                "value": "+",
                "operator": "=="
            },
            "Only identified by site": {
                "value": "+",
                "operator": "=="
            },
            "Potential contaminant": {
                "value": "+",
                "operator": "=="
            },
        }
        operation_keep = {
            "Razor + unique peptides": {
                "value": 2,
                "operator": ">="
            }
        }
        self.da.remove_rows(operation_remove)
        self.da.remove_rows(operation_keep, True)

        assert (
                (self.da.current_df["Reverse"] != "+") &
                (self.da.current_df["Only identified by site"] != "+") &
                (self.da.current_df["Potential contaminant"] != "+")
        ).all()
        assert (self.da.current_df["Razor + unique peptides"] >= 2).all()

    def test_impute_lcm(self):
        self.test_impute_missing()
        self.da.impute_lcm([e.name for e in self.experiments], 1.8, 0.3, self.conditions, ">", 0)
        assert pd.notnull(self.da.current_df).all().all()

    def test_imput_missing_forest(self):
        self.test_impute_missing()
        self.da.impute_missing_forest([e.name for e in self.experiments], self.conditions, ">", 0)
        assert pd.notnull(self.da.current_df).all().all()

    def test_print_procedure(self):
        self.test_impute_missing()
        self.da.impute_missing_forest([e.name for e in self.experiments], self.conditions, ">", 0)
        self.da.print_procedure()

    def test_impute_missing(self):
        self.test_get_columns()
        self.test_add_columns()
        self.test_remove_rows()
        self.da.impute_missing([e.name for e in self.experiments], self.conditions, 3, 1)

    def test_rewind(self):
        self.test_get_columns()
        self.test_add_columns()
        self.test_remove_rows()
        a = self.da.impute_missing([e.name for e in self.experiments], self.conditions, 3, 1, True)
        b = a.rewind(2)
        assert b[0].current_df_position == 7

    def test_forward(self):
        self.test_get_columns()
        self.test_add_columns()
        self.test_remove_rows()
        a = self.da.impute_missing([e.name for e in self.experiments], self.conditions, 3, 1, True)
        b, c = a.rewind(2)
        d, f = b.forward(1)

        g = d.get_all_operations()
        assert b.current_df_position == 8

    def test_normalize_median(self):
        self.test_impute_lcm()
        self.da.normalize([e.name for e in self.experiments], method="median")
        assert len(np.unique(self.da.current_df.median(axis=0))) == 1

    def test_normalize_mean(self):
        self.test_impute_lcm()
        self.da.normalize([e.name for e in self.experiments], method="mean")
        assert len(np.unique(self.da.current_df.median(axis=0))) == 1

    def test_normalize_zscore(self):
        self.test_impute_lcm()
        self.da.normalize([e.name for e in self.experiments], method="z-score")

    def test_ttest(self):
        self.test_normalize_zscore()
        self.da.two_sample([["A", "B"]], False, self.conditions, [e.name for e in self.experiments])

    def test_anova(self):
        self.test_normalize_zscore()
        self.da.anova(self.conditions, self.conditions, [e.name for e in self.experiments])

    def test_correction(self):
        self.test_anova()
        self.da.p_correct(0.05, "fdr_bh")

    def test_volcano(self):
        self.test_normalize_zscore()
        self.da.two_sample([["A", "B"]], False, self.conditions, [e.name for e in self.experiments])
        self.da.p_correct(0.05, "fdr_bh")
        self.da.volcano_plot(
            0.01, 0.5,
            display_text=True,
            title="A vs B",
            # filename=r"C:\Users\toanp\OneDrive\other docs\GitHub\Pyxis\tests\AvsB.png"
        )

    def test_fuzzy_c(self):
        self.test_normalize_zscore()
        self.da.fuzzy_c([e.name for e in self.experiments],
                        self.conditions,
                        # filename=r"C:\Users\toanp\OneDrive\other docs\GitHub\Pyxis\tests\AvsB.png"
                        )
        # self.da.plot.draw()

    def test_box_plot(self):
        self.test_normalize_zscore()
        self.da.box_plot(
            [e.name for e in self.experiments],
            self.conditions
        )

    def test_box_plot_fuzzy(self):
        self.test_fuzzy_c()
        self.da.add_columns([self.da.df["Gene names"]])
        a = self.da.current_df.reset_index()
        d = []
        for i, r in a[[a.columns[0]] + ["Gene names"]].iterrows():
            d.append(f"{r['Gene names'].split(';')[0]}  ({r[a.columns[0]].split(';')[0]})")

        d = pd.Series(d, index=self.da.current_df.index)

        d.name = "ID"

        self.da.add_columns([d])
        self.da.box_plot(
            [e.name for e in self.experiments],
            self.conditions,
            by_label=True,
            label_col="ID",
            group_col="Cluster",
            discrete_colors=self.analysis.discrete_colors,
            filename="testCluster.png"
        )

    def test_rank_plot(self):
        self.test_ttest()
        self.da.p_correct(0.05, "fdr_bh")
        self.da.add_columns([self.da.df["Gene names"]])
        a = self.da.current_df.reset_index()
        d = []
        for i, r in a[[a.columns[0]] + ["Gene names"]].iterrows():
            d.append(f"{r['Gene names'].split(';')[0]}  ({r[a.columns[0]].split(';')[0]})")

        d = pd.Series(d, index=self.da.current_df.index)

        d.name = "ID"
        self.da.add_columns([d])
        self.da.rank_plot([e.name for e in self.experiments], self.conditions, "A", "B", display_text=True,
                          text_column="ID", discrete_colors=self.analysis.discrete_colors)

    def test_go_enrich(self):
        self.test_ttest()
        self.da.p_correct(0.05, "fdr_bh")
        self.da.current_df.reset_index(inplace=True)
        self.da.go_terms_enrichment(self.da.current_df[self.da.current_df["adj.p-value"] < 0.2], "Majority protein IDs",
                                    "Majority protein IDs", ["fdr_bh"], 0.1, filename="test.png")

    def test_correlation_heatmap(self):
        self.test_imput_missing_forest()
        self.da.correlation_heatmap([e.name for e in self.experiments], self.conditions)

    def test_correlation_scatter(self):
        self.test_imput_missing_forest()
        self.da.correlation_scatter([e.name for e in self.experiments], self.conditions)

    def test_set_df_as_current(self):
        self.test_get_columns()
        self.test_add_columns()
        self.test_remove_rows()
        self.da.impute_missing([e.name for e in self.experiments], self.conditions, 3, 1, True)
        self.da.set_df_as_current(3)
        self.da.print_procedure()

    def test_delete(self):
        self.test_get_columns()
        self.test_add_columns()
        self.test_remove_rows()
        self.da.impute_missing([e.name for e in self.experiments], self.conditions, 3, 1, True)
        self.da.set_df_as_current(1)
        self.da.print_procedure()
        self.da.delete(5)
        self.da.print_procedure()

    def test_normalize_quantile(self):
        self.test_impute_lcm()
        self.da.normalize([e.name for e in self.experiments], method="quantile")

    def test_limma(self):
        self.test_normalize_quantile()
        self.da.limma([["A", "B"]], self.conditions, [e.name for e in self.experiments])
        self.da.current_df.to_csv("test.csv")

class TestMetabolomics(TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.da = Data(file_path=r"C:\Users\toanp\Downloads\data (36).csv", index="PG.ProteinGroups", history=True)
        a = ["IP_RC_Hom", "IP_RC_Het", "IP_WT", "IP_Mock", "WCL_RC_Hom", "WCL_RC_Het", "WCL_RC_WT", "WCL_Mock"]
        r = [3, 3, 3, 3, 3, 3, 3, 3]
        self.experiments = []
        self.conditions = []
        for i in range(len(a)):
            for i2 in range(r[i]):
                self.experiments.append(f"{self.conditions}.{i2 + 1}")
                self.conditions.append(a[i])
        self.da.current_df.columns = self.experiments
        #
        # self.experiments = []
        # experiment_dict = {}
        # for c in self.da.current_df.columns:
        #     if c.startswith("Intensity "):
        #         e = Experiment(c)
        #         experiment_dict[int(e.replicate)] = e
        # a = list(experiment_dict.keys())
        # a = sorted(a)
        # for c in a:
        #     self.experiments.append(experiment_dict[c])
        # self.conditions = ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "D", "D", "D", "E", "E"]

    def test_stuff(self):
        self.da.impute_missing_forest(self.experiments, self.conditions, ">", 0)
        self.da.fuzzy_c(self.experiments, self.conditions)
        self.da.current_df.to_csv("test.csv")
