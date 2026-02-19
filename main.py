import flask

app = flask.Flask(__name__)

@app.route("/")
def home():
    return flask.render_template("index.html")

@app.route("/<path:path>")
def dynamic(path):
    return flask.render_template(
        "index.html",
        path=path,
        args=flask.request.args
    )

@app.route("/cookie")
def cookie():
    user_cookie = flask.request.cookies.get("subscribe")
    if user_cookie is None:
        resp = flask.make_response(flask.redirect("/"))
        resp.set_cookie("subscribe", "done")
        return resp
    else:
        return flask.redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
