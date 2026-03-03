import nodemailer from 'nodemailer';

// Create transporter with Gmail credentials
const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: process.env.EMAIL_USER || 'cn3581743@gmail.com',
    pass: process.env.EMAIL_PASS || 'jyag rnwm ezvo yqlb'
  }
});

// Generate 6-digit OTP
export const generateOTP = () => {
  return Math.floor(100000 + Math.random() * 900000).toString();
};

// Send OTP email
export const sendOTPEmail = async (email, otp, userType) => {
  const mailOptions = {
    from: {
      name: 'MediConsult - Zero Trust Secure Telehealth Platform',
      address: process.env.EMAIL_USER || 'cn3581743@gmail.com'
    },
    to: email,
    subject: 'Email Verification - MediConsult OTP',
    html: `
      <!DOCTYPE html>
      <html>
      <head>
        <style>
          body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
          }
          .container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 30px;
            color: white;
          }
          .otp-box {
            background: white;
            color: #667eea;
            font-size: 32px;
            font-weight: bold;
            letter-spacing: 8px;
            text-align: center;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
          }
          .info {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
          }
          .footer {
            text-align: center;
            margin-top: 20px;
            font-size: 12px;
            color: rgba(255, 255, 255, 0.8);
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h2 style="margin-top: 0;">🏥 MediConsult Email Verification</h2>
          <p>Hello,</p>
          <p>Thank you for registering as a <strong>${userType === 'doctor' ? 'Doctor' : 'Patient'}</strong> on <strong>MediConsult</strong> - Zero Trust Secure Telehealth Platform.</p>
          <p>Your One-Time Password (OTP) for email verification is:</p>

          <div class="otp-box">${otp}</div>

          <div class="info">
            <p style="margin: 5px 0;"><strong>⏰ Valid for:</strong> 10 minutes</p>
            <p style="margin: 5px 0;"><strong>🔒 Security:</strong> Do not share this OTP with anyone</p>
            <p style="margin: 5px 0;"><strong>❓ Not you?</strong> Please ignore this email</p>
          </div>

          <p style="margin-top: 20px;">If you didn't request this verification, please ignore this email.</p>

          <div class="footer">
            <p>© 2024 MediConsult - Zero Trust Secure Telehealth Platform</p>
            <p>Secure • Biometric • Trusted</p>
          </div>
        </div>
      </body>
      </html>
    `
  };

  try {
    const info = await transporter.sendMail(mailOptions);
    console.log('OTP Email sent:', info.messageId);
    return { success: true, messageId: info.messageId };
  } catch (error) {
    console.error('Error sending OTP email:', error);
    throw new Error('Failed to send OTP email');
  }
};

// Verify transporter configuration
export const verifyEmailConfig = async () => {
  try {
    await transporter.verify();
    console.log('✅ Email service is ready to send emails');
    return true;
  } catch (error) {
    console.error('❌ Email service configuration error:', error);
    return false;
  }
};

