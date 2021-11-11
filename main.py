import asyncio
import os
import sys
import tornado.httpserver
from tornado.ioloop import IOLoop
from tornado.web import Application
from tornado.options import define, options

from adept.handlers import MainHandler, AnalysisWebSocket

if not os.environ.get("R_LIB_LOC"):
    define("r_lib_loc", default="C:/Users/toanp/OneDrive/other docs/adept/renv/library/R-4.0/x86_64-w64-mingw32",
           help="Location of R library")
else:
    define("r_lib_loc", default=os.environ.get("R_LIB_LOC"),
           help="Location of R library")

define("port", default=8000, help="Port number")


analysis_cache = {}

if sys.platform.startswith("win32"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if __name__ == '__main__':
    tornado.options.parse_command_line()
    app = Application()
    app.add_handlers(
        r'(localhost|127\.0\.0\.1|adept\.proteo\.info|10\.202\.62\.27)',
        [
            (r"/", MainHandler),
            (r"/rpc", AnalysisWebSocket)
            # (r"/static", StaticFileHandler, dict(path=settings['static_path']))
        ]
    )
    server = tornado.httpserver.HTTPServer(app)
    server.bind(options.port)
    os.environ["R_LIB_LOC"] = options.r_lib_loc
    server.start(1)
    IOLoop.current().start()
