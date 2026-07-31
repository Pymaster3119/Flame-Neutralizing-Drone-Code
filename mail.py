import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SENDER_EMAIL = "cygnus3119@gmail.com"
APP_PASSWORD = "yojx bpic qqjq ufcx"
RECEIVER_EMAIL = "aditya.anand3119@gmail.com"

# email
message = MIMEMultipart()
message["From"] = SENDER_EMAIL
message["To"] = RECEIVER_EMAIL
message["Subject"] = "WOW A FIRE!!!"

body = "yo bro theres a fire in ur house"
message.attach(MIMEText(body, "plain"))

try:
    # Connect to Gmail's SMTP server using port 587 (TLS)y
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()  # Secure and encrypt the connection
    
    # Log in and send email
    server.login(SENDER_EMAIL, APP_PASSWORD)
    server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, message.as_string())
    
    print("sent")

except Exception as e:
    print(e)

finally:
    server.quit()