import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import URLGenerator

def SendEmail(hyperlink):
    sender_email = 'cygnus3119@gmail.com'
    password = 'elgu hobj hkyu nuba'
    receiver_email = 'incognitomode2008@gmail.com'

    message = MIMEMultipart("alternative")
    message["From"] = sender_email
    message['To'] = receiver_email
    message['Subject'] = "Fire Detected"

    with open("EmailSystem.html","r",encoding="utf-8") as file:
        EmailSystem = file.read()

    EmailSystem = EmailSystem.replace("{{ input link }}", hyperlink) 
    body = EmailSystem



    message.attach(MIMEText(body, "html"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()

        server.login(sender_email, password)

        server.sendmail(sender_email, receiver_email, message.as_string())

        print("Sent!")
    except Exception as e:
        print(e)
        print("Failed")

    finally:
        server.quit()
        
if __name__ == "__main__":
    SendEmail(URLGenerator.CreateIMGBBURL("/home/gset/Desktop/img.jpg", "d7f80c64db14611e7c860157242e0e93"))