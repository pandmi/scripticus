import smtplib, ssl
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.message import EmailMessage           
from email.mime.application import MIMEApplication
import email
import email.mime.application
from email.mime.base import MIMEBase

from os.path import basename, expanduser
from email import encoders as Encoders
from email.utils import formatdate
from email.mime.text import MIMEText

def send_email(msg_from=None, msg_to=None, msg_cc=None, msg_bcc=None, subject=None, body='', attachments=None,connection_settings=None,noreply=False,footer=True, password=None):
    def make_list(text_or_list):
        if isinstance(text_or_list, list):
            return text_or_list
        else:
            return [text_or_list]
    # Build an email
    msg = MIMEMultipart('alternative')
    msg['From'] = msg_from
    CONF_FOOTER = """
----
Confidentiality Notice:
This email and any attachments to it are intended only for use by the addressee.  This email and any attachments to it may contain information that is confidential, legally privileged and exempt from disclosure.  If you are not the addressee, or the person responsible for delivering this email and any attachments to it to the addressee, you are hereby notified that any disclosure, copying, distribution or use of the email and any attachments to it is strictly prohibited.  If you have received this transmission in error, please contact the sender and delete the email, including any attachments to it, from your computer.  Thank you."""
    
    msg['Subject'] = subject
    # msg.attach(MIMEText(body + CONF_FOOTER))
    msg.attach(MIMEText(body))
    
    if msg_from:
        msg['From'] = ','.join(make_list(msg_from))
    if msg_to:
        msg['To'] = ','.join(make_list(msg_to))
    if msg_cc:
        msg['Cc'] = ','.join(make_list(msg_cc))
    if msg_bcc:
        msg['Bcc'] = ','.join(make_list(msg_bcc))
    
    # What a recipient sees if they don't use an email reader
    msg.preamble = 'Multipart message.\n'
    
    if noreply:
        msg.add_header('reply-to', 'noreply@mediamath.com')

    msg['Date'] = formatdate(localtime=True)
    
    
    if attachments:
        attachments = make_list(attachments)
        for attachment in attachments:
            with open(expanduser(attachment), 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
            Encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment; filename='
                            '"{}"'.format(basename(attachment)))
            msg.attach(part)

    context=ssl.create_default_context()

    with smtplib.SMTP("smtp.office365.com", port=587) as smtp:
        smtp.starttls(context=context)
        smtp.login(msg["From"], password)
        smtp.send_message(msg)




# # Create message container - the correct MIME type is multipart/alternative.

 
# # Create the body of the message (a plain-text and an HTML version).
# css = css_file.read()
# html_body = report_file.read()
# html = '<style type="text/css">'+css+'</style>'+html_body


# # Record the MIME types of both parts - text/plain and text/html.
# # part1 = MIMEText(css, 'plain')
# part2 = MIMEText(html, 'html')
 
#         # create PDF attachment
# filename='Havas_Spain_campaign_monitoring_Repsol.html'
# fp=open(filename,'rb')
# att = email.mime.application.MIMEApplication(fp.read(),_subtype="html")
# fp.close()
# att.add_header('Content-Disposition','attachment',filename=filename)

# # Attach parts into message container.
# msg.attach(att)


# filename_2='custom.css'
# fp_2=open(filename_2,'rb')
# att_2 = email.mime.application.MIMEApplication(fp_2.read(),_subtype="css")
# fp_2.close()
# att_2.add_header('Content-Disposition','attachment',filename=filename_2)

# # Attach parts into message container.
# msg.attach(att_2)



# msg.attach(part2)
# context=ssl.create_default_context()

# with smtplib.SMTP("smtp.office365.com", port=587) as smtp:
#     smtp.starttls(context=context)
#     smtp.login(msg["From"], "RosaHutor2019!")
#     smtp.send_message(msg)
    