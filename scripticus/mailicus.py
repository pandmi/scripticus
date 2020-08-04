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


