from io import StringIO
from uuid import uuid4

import pandas as pd
from tornado.escape import json_decode
from tornado.web import RequestHandler
from tornado.websocket import WebSocketHandler

from main import analysis_cache
from pyxis.Pyxis import Analysis, Data


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
        if data["message"] == "request-id":
            unique_id = str(uuid4())
            analysis_cache[unique_id] = {"analysis": {}, "df": pd.DataFrame(), "settings": {}}
            self.write_message({"id": unique_id, "origin": "request-id"})
        elif data["message"] == "upload-starter":
            analysis_cache[data["id"]]["df"] = pd.read_csv(StringIO(data["data"]), sep="\t")

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
    def on_close(self):
        pass

    def check_origin(self, origin: str) -> bool:
        if origin == "http://localhost:4200":
            return True