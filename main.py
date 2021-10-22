import asyncio
import sys
import tornado.httpserver
from tornado.ioloop import IOLoop
from tornado.web import Application
from tornado.options import define, options

from adept.handlers import MainHandler, AnalysisWebSocket

define("port", default=8000, help="Port number")

analysis_cache = {}

if sys.platform.startswith("win32"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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
