from flask import Flask
from flask_mail import Mail, Message

app = Flask(__name__)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'orabyabdo21@gmail.com'
app.config['MAIL_PASSWORD'] = 'pwlc uwbq wuqj jwsj'

mail = Mail(app)

@app.route("/")
def send_test():
    msg = Message(
        subject="Test Email",
        sender=app.config['MAIL_USERNAME'],
        recipients=["orabyabdo21@gmail.com"],
        body="Hello from Flask"
    )
    mail.send(msg)
    return "Email Sent!"

if __name__ == "__main__":
    app.run(debug=True)