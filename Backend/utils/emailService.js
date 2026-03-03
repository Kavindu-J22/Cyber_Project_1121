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

// Send appointment approval email
export const sendAppointmentApprovalEmail = async (patientEmail, appointmentData) => {
  const { appointmentNumber, doctorName, appointmentDate, appointmentTimeFrom, appointmentTimeTo, doctorNote } = appointmentData;

  const mailOptions = {
    from: {
      name: 'MediConsult - Zero Trust Secure Telehealth Platform',
      address: process.env.EMAIL_USER || 'cn3581743@gmail.com'
    },
    to: patientEmail,
    subject: `Appointment Approved - ${appointmentNumber}`,
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
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            border-radius: 10px;
            padding: 30px;
            color: white;
          }
          .status-badge {
            background: white;
            color: #10b981;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
          }
          .details {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
          }
          .detail-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
          }
          .detail-row:last-child {
            border-bottom: none;
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
          <h2 style="margin-top: 0;">✅ Appointment Approved</h2>
          <p>Dear Patient,</p>
          <p>Great news! Your appointment request has been <strong>approved</strong> by Dr. ${doctorName}.</p>

          <div class="status-badge">
            Appointment Number: ${appointmentNumber}
          </div>

          <div class="details">
            <h3 style="margin-top: 0;">Appointment Details</h3>
            <div class="detail-row">
              <strong>Doctor:</strong>
              <span>Dr. ${doctorName}</span>
            </div>
            <div class="detail-row">
              <strong>Date:</strong>
              <span>${new Date(appointmentDate).toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</span>
            </div>
            <div class="detail-row">
              <strong>Time:</strong>
              <span>${appointmentTimeFrom} - ${appointmentTimeTo}</span>
            </div>
            ${doctorNote ? `
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255, 255, 255, 0.2);">
              <strong>Doctor's Note:</strong>
              <p style="margin: 10px 0 0 0;">${doctorNote}</p>
            </div>
            ` : ''}
          </div>

          <p style="margin-top: 20px;">Please make sure to arrive on time for your appointment.</p>

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
    console.log('Appointment approval email sent:', info.messageId);
    return { success: true, messageId: info.messageId };
  } catch (error) {
    console.error('Error sending appointment approval email:', error);
    throw new Error('Failed to send appointment approval email');
  }
};

// Send appointment rejection email
export const sendAppointmentRejectionEmail = async (patientEmail, appointmentData) => {
  const { appointmentNumber, doctorName, reason, doctorNote } = appointmentData;

  const mailOptions = {
    from: {
      name: 'MediConsult - Zero Trust Secure Telehealth Platform',
      address: process.env.EMAIL_USER || 'cn3581743@gmail.com'
    },
    to: patientEmail,
    subject: `Appointment Update - ${appointmentNumber}`,
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
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            border-radius: 10px;
            padding: 30px;
            color: white;
          }
          .status-badge {
            background: white;
            color: #ef4444;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
          }
          .details {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 20px;
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
          <h2 style="margin-top: 0;">❌ Appointment Update</h2>
          <p>Dear Patient,</p>
          <p>We regret to inform you that your appointment request has been declined by Dr. ${doctorName}.</p>

          <div class="status-badge">
            Appointment Number: ${appointmentNumber}
          </div>

          <div class="details">
            <h3 style="margin-top: 0;">Request Details</h3>
            <p><strong>Doctor:</strong> Dr. ${doctorName}</p>
            <p><strong>Reason for Request:</strong> ${reason}</p>
            ${doctorNote ? `
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255, 255, 255, 0.2);">
              <strong>Doctor's Note:</strong>
              <p style="margin: 10px 0 0 0;">${doctorNote}</p>
            </div>
            ` : ''}
          </div>

          <p style="margin-top: 20px;">You may submit a new appointment request or contact our support team for assistance.</p>

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
    console.log('Appointment rejection email sent:', info.messageId);
    return { success: true, messageId: info.messageId };
  } catch (error) {
    console.error('Error sending appointment rejection email:', error);
    throw new Error('Failed to send appointment rejection email');
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

