import qrcode

def generate_qr(url: str, output_file: str = 'qrcode.png') -> None:
    """
    Generates a QR code for the given URL and saves it as an image file.

    Args:
        url (str): The URL to encode in the QR code.
        output_file (str): The filename for the saved QR code image.
    """
    # Create qr code instance
    qr = qrcode.QRCode(
        version=1,  # controls the size of the QR Code
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    # Create an image from the QR Code instance
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(output_file)
    print(f"QR code generated and saved as {output_file}")

if __name__ == "__main__":
    url = input("Enter the URL to generate QR code: ")
    generate_qr(url) 
