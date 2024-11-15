import smtplib
from email.message import EmailMessage
from email.headerregistry import Address


def send_failure_email(subject, content, addresses_to=None, address_from=None):
    """
    This function sends emails notification to notify given recipients of a failure in the data generation
    of PAGER pipeline modules. Subject, content and addresses needs to be accurately chosen. An address check will
    be developed later on.

    :param subject: The subject of the notification. Should be short, containing the work package of reference,
                    and the module of reference. Example  "WP2 SWIFT OUTPUT FAILURE"
    :type subject: str
    :param content: The content of the message describing the reason for the email being sent
    :type content: str
    :param addresses_to: tuple of addresses to which the email notification is sent.
    :type addresses_to: tuple
    :param address_from: The address from where to send the email notification. If None, a default will be used.
    :type address_from: email.headerregistry.Address
    """
    msg = EmailMessage()
    msg['Subject'] = subject
    if address_from is None:
        msg['From'] = Address("spacepager", "spacepager", "sec27.de")
    else:
        assert isinstance(address_from, Address), "Sending address {} not in correct format...".format(address_from)
        msg['From'] = address_from
    for address in addresses_to:
        assert isinstance(address, Address), "Recipient address {} not in correct format...".format(address)
    msg['To'] = addresses_to
    msg.set_content(content)

    # Send the message via local SMTP server.
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()
