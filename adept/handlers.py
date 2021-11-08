from io import StringIO
from uuid import uuid4

import pandas as pd
from tornado.escape import json_decode
from tornado.web import RequestHandler
from tornado.websocket import WebSocketHandler
from pyxis.Pyxis import Analysis, Data
import pickle
import os
analysis_cache = {}

static_loc = "static"

class BaseHandler(RequestHandler):
    def set_default_headers(self):
        self.set_header("access-control-allow-origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'GET, PUT, DELETE, OPTIONS')
        self.set_header("Access-Control-Allow-Headers", "access-control-allow-origin,authorization,content-type")

    def options(self):
        self.set_status(204)
        self.finish()


class MainHandler(BaseHandler):
    def get(self):
        self.write("Hello, world")


class RequestID(BaseHandler):
    def post(self):
        data = json_decode(self.request.body)


class AnalysisWebSocket(WebSocketHandler):
    def open(self):
        self.write_message({"message": "connected"})

    def on_message(self, message):
        data = json_decode(message)
        if data["position"] != 0:
            if data["id"] in analysis_cache:
                if data["position"] - 1 <= analysis_cache[data["id"]]["analysis"].data.current_df_position:
                    analysis_cache[data["id"]]["analysis"].data.delete(data["position"])

        if data["message"] == "request-id":
            unique_id = str(uuid4())
            analysis_cache[unique_id] = {"analysis": {}, "df": pd.DataFrame(), "settings": {}}
            self.write_message({"id": unique_id, "origin": "request-id"})
        elif data["message"] == "upload-starter":
            analysis_cache[data["id"]]["df"] = pd.read_json(StringIO(data["data"]))
            analysis_cache[data["id"]]["settings"] = json_decode(data["settings"])
            analysis_cache[data["id"]]["df"] = analysis_cache[data["id"]]["df"].set_index(
                analysis_cache[data["id"]]["settings"]["primaryIDColumns"])
            experiments = []
            conditions = []
            for a in analysis_cache[data["id"]]["settings"]["experiments"]:
                experiments.append(a["name"])
                conditions.append(a["condition"])
            analysis_cache[data["id"]]["analysis"] = Analysis(conditions)
            analysis_cache[data["id"]]["analysis"].experiments = experiments
            analysis_cache[data["id"]]["analysis"].data = Data(analysis_cache[data["id"]]["df"], history=True)
            self.write_message({"id": data["id"], "origin": "upload-starter",
                                "data": analysis_cache[data["id"]]["analysis"].data.current_df.to_csv(sep="\t")})
        elif data["message"] in ["Random Forest", "Left Censored Median", "Simple"]:
            parameters = json_decode(data["data"])
            if data["message"] == "Random Forest":
                analysis_cache[data["id"]]["analysis"].data.impute_missing_forest(
                    analysis_cache[data["id"]]["analysis"].experiments,
                    analysis_cache[data["id"]]["analysis"].conditions,
                    ">",
                    parameters["# missing values/condition"]
                )
            elif data["message"] == "Left Censored Median":
                analysis_cache[data["id"]]["analysis"].data.impute_lcm(
                    analysis_cache[data["id"]]["analysis"].experiments,
                    data["Down-shift"],
                    data["Width"],
                    analysis_cache[data["id"]]["analysis"].conditions,
                    ">",
                    parameters["# missing values/condition"]
                )
            elif data["message"] == "Simple":
                analysis_cache[data["id"]]["analysis"].data.impute_missing(
                    analysis_cache[data["id"]]["analysis"].experiments,
                    analysis_cache[data["id"]]["analysis"].conditions,
                    data["# missing values/condition"],
                    data["# missing values/row"],
                )
            self.write_message({"id": data["id"], "origin": "imputation",
                                "data": analysis_cache[data["id"]]["analysis"].data.current_df.to_csv(sep="\t")})
        elif data["message"] == "Normalization":
            if data["data"] == "Median":
                analysis_cache[data["id"]]["analysis"].data.normalize(
                    analysis_cache[data["id"]]["analysis"].experiments, method="median"
                )
            elif data["data"] == "Mean":
                analysis_cache[data["id"]]["analysis"].data.normalize(
                    analysis_cache[data["id"]]["analysis"].experiments, method="mean"
                )
            elif data["data"] == "Z-Score Row":
                analysis_cache[data["id"]]["analysis"].data.normalize(
                    analysis_cache[data["id"]]["analysis"].experiments, method="z-score"
                )
            elif data["data"] == "Z-Score Column":
                analysis_cache[data["id"]]["analysis"].data.normalize(
                    analysis_cache[data["id"]]["analysis"].experiments, method="z-score-col"
                )
            self.write_message({"id": data["id"], "origin": "normalization",
                                "data": analysis_cache[data["id"]]["analysis"].data.current_df.to_csv(sep="\t")})
        elif data["message"] == "TTest":
            comparisons = json_decode(data["data"])
            analysis_cache[data["id"]]["analysis"].data.two_sample(
                [[c["A"], c["B"]] for c in comparisons],
                False,
                analysis_cache[data["id"]]["analysis"].conditions,
                analysis_cache[data["id"]]["analysis"].experiments
            )
            self.write_message({"id": data["id"], "origin": "ttest",
                                "data": analysis_cache[data["id"]]["analysis"].data.current_df.to_csv(sep="\t")})
        elif data["message"] in ["bonferonni", "sidak", "holm-sidak", "holm", "simes-hochberg", "hommel", "fdr_bh",
                                 "fdr_by", "fdr_tsbh", "fdr_tsbky"]:
            analysis_cache[data["id"]]["analysis"].data.p_correct(
                float(data["data"]),
                data["message"]
            )
            self.write_message({"id": data["id"], "origin": "p-correct",
                                "data": analysis_cache[data["id"]]["analysis"].data.current_df.to_csv(sep="\t")})
        elif data["message"] == "ANOVA":
            selected = json_decode(data["data"])
            analysis_cache[data["id"]]["analysis"].data.anova(
                selected,
                analysis_cache[data["id"]]["analysis"].conditions,
                analysis_cache[data["id"]]["analysis"].experiments
            )
            self.write_message({"id": data["id"], "origin": "anova",
                                "data": analysis_cache[data["id"]]["analysis"].data.current_df.to_csv(sep="\t")})
        elif data["message"] == "Filter":
            filter_steps = json_decode(data["data"])
            keep = {}
            remove = {}
            for f in filter_steps:
                if f["valueType"] == "number":
                    f["value"] = float(f["value"])
                if f["keep"]:
                    keep[f["column"]] = {"value": f["value"], "operator": f["operator"]}
                else:
                    remove[f["column"]] = {"value": f["value"], "operator": f["operator"]}
            if keep:
                analysis_cache[data["id"]]["analysis"].data.remove_rows(
                    keep,
                    keep=True
                )
            if remove:
                analysis_cache[data["id"]]["analysis"].data.remove_rows(
                    remove,
                )
            self.write_message({"id": data["id"], "origin": "filter",
                                "data": analysis_cache[data["id"]]["analysis"].data.current_df.to_csv(sep="\t")})
        elif data["message"] == "Fuzzy":
            analysis_cache[data["id"]]["analysis"].data.fuzzy_c(
                analysis_cache[data["id"]]["analysis"].experiments,
                analysis_cache[data["id"]]["analysis"].conditions,
                float(data["data"])
            )
            self.write_message({"id": data["id"], "origin": "fuzzy",
                                "data": analysis_cache[data["id"]]["analysis"].data.current_df.to_csv(sep="\t")})
        elif data["message"] == "ChangeCurrentDF":
            analysis_cache[data["id"]]["analysis"].data.set_df_as_current(int(data["data"]))
            self.write_message({"id": data["id"], "origin": "changeCurrentDf",
                                "data": analysis_cache[data["id"]]["analysis"].data.current_df.to_csv(sep="\t")})

        elif data["message"] == "EndSession":
            if data["id"] in analysis_cache:
                del analysis_cache[data["id"]]

        elif data["message"] == "DeleteNode":
            if data["id"] in analysis_cache:
                if int(data["data"]) - 1 <= analysis_cache[data["id"]]["analysis"].data.current_df_position:
                    analysis_cache[data["id"]]["analysis"].data.delete(int(data["data"]))
                    self.write_message({"id": data["id"], "origin": "deleteNode",
                                        "data": analysis_cache[data["id"]]["analysis"].data.current_df.to_csv(
                                            sep="\t")})

        elif data["message"] == "CorrelationMatrix":
            if data["id"] in analysis_cache:
                analysis_cache[data["id"]]["analysis"].data.correlation_heatmap(
                    analysis_cache[data["id"]]["analysis"].experiments,
                    analysis_cache[data["id"]]["analysis"].conditions,
                )

                self.write_message({"id": data["id"], "origin": "correlationMatrix",
                                    "data": analysis_cache[data["id"]]["analysis"].data.current_df.to_csv(sep="\t")})
        elif data["message"] == "LoadSaved":
            if data["data"]:
                with open(os.path.join(static_loc, data["data"]), "rb") as infile:
                    unique_id = str(uuid4())
                    analysis_cache[str(unique_id)] = pickle.load(infile)
                    print(analysis_cache[str(unique_id)])
                    self.write_message({"id": data["id"], "origin": "loadSaved",
                                        "data": analysis_cache[data["id"]]["json"]})

        elif data["message"] == "SaveAnalysis":
            if data["id"] in analysis_cache:
                with open(os.path.join(static_loc, data["id"]), "wb") as output:
                    analysis_cache[data["id"]]["json"] = data["data"]
                    pickle.dump(analysis_cache[data["id"]], output)

                    self.write_message({"id": data["id"], "origin": "saveAnalysis",
                        "data": data["id"]})


    def on_close(self):
        pass

    def check_origin(self, origin: str) -> bool:
        if origin == "http://localhost:4200" or origin == "http://adept.proteo.info":
            return True
