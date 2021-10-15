
import asyncio
import os
import sys
from ctypes import Union
from io import StringIO
from uuid import uuid4
import pandas as pd
from tornado.escape import json_decode
from tornado.websocket import WebSocketHandler
import tornado.httpserver
from tornado.ioloop import IOLoop
from tornado.web import Application, StaticFileHandler, RequestHandler
from tornado.options import define, options

from pyxis.Pyxis import Analysis, Data

define("port", default=8000, help="Port number")


analysis_cache = {}

if sys.platform.startswith("win32"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


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
            analysis_cache[data["id"]]["df"] = analysis_cache[data["id"]]["df"].set_index(analysis_cache[data["id"]]["settings"]["primaryIDColumns"])
            experiments = []
            conditions = []
            for a in analysis_cache[data["id"]]["settings"]["experiments"]:
                experiments.append(a["name"])
                conditions.append(a["condition"])
            analysis_cache[data["id"]]["analysis"] = Analysis(conditions)
            analysis_cache[data["id"]]["analysis"].experiments = experiments
            analysis_cache[data["id"]]["analysis"].data = Data(analysis_cache[data["id"]]["df"], history=True)
            self.write_message({"id": data["id"], "origin": "upload-starter", "data": analysis_cache[data["id"]]["analysis"].data.current_df.to_csv(sep="\t")})



    def on_close(self):
        pass

    def check_origin(self, origin: str) -> bool:
        return True

if __name__ == '__main__':
    tornado.options.parse_command_line()
    app = Application()
    app.add_handlers(
        r'(localhost|127\.0\.0\.1|adept\.proteo\.info)',
        [
            (r"/", MainHandler),
            (r"/rpc", AnalysisWebSocket)
            # (r"/static", StaticFileHandler, dict(path=settings['static_path']))
        ]
    )
    server = tornado.httpserver.HTTPServer(app)
    server.bind(options.port)

    server.start(1)
    IOLoop.current().start()

