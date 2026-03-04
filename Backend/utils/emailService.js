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
    subject: userType === 'consultation' ? '🔐 Consultation Security OTP - MediConsult' : 'Email Verification - MediConsult OTP',
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
          <h2 style="margin-top: 0;">${userType === 'consultation' ? '🔐 Consultation Security Verification' : '🏥 MediConsult Email Verification'}</h2>
          <p>Hello,</p>
          ${userType === 'consultation'
            ? '<p>You are about to start a <strong>secure consultation session</strong> on <strong>MediConsult</strong>. Please use the OTP below to verify your identity before proceeding.</p>'
            : `<p>Thank you for registering as a <strong>${userType === 'doctor' ? 'Doctor' : 'Patient'}</strong> on <strong>MediConsult</strong> - Zero Trust Secure Telehealth Platform.</p>`
          }
          <p>Your One-Time Password (OTP) for ${userType === 'consultation' ? 'consultation start verification' : 'email verification'} is:</p>

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

// Send consultation started notification to patient
export const sendConsultationStartedEmail = async (patientEmail, patientName, doctorName, appointmentNumber, scheduledDate, scheduledTime) => {
  const mailOptions = {
    from: {
      name: 'MediConsult - Zero Trust Secure Telehealth Platform',
      address: process.env.EMAIL_USER || 'cn3581743@gmail.com'
    },
    to: patientEmail,
    subject: `🎥 Consultation Started - ${appointmentNumber}`,
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
          .content {
            background: white;
            color: #333;
            border-radius: 8px;
            padding: 25px;
            margin-top: 20px;
          }
          .header {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
          }
          .info-box {
            background: #f0fdf4;
            border-left: 4px solid #10b981;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
          }
          .button {
            display: inline-block;
            background: #10b981;
            color: white;
            padding: 12px 30px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            margin-top: 15px;
          }
          .footer {
            margin-top: 20px;
            font-size: 12px;
            opacity: 0.9;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">🎥 Consultation Started!</div>
          <p>Your doctor is ready for the consultation</p>

          <div class="content">
            <p><strong>Dear ${patientName},</strong></p>

            <p>Great news! Dr. ${doctorName} has started the consultation session and is waiting for you to join.</p>

            <div class="info-box">
              <p style="margin: 5px 0;"><strong>📋 Appointment Number:</strong> ${appointmentNumber}</p>
              <p style="margin: 5px 0;"><strong>📅 Scheduled Date:</strong> ${scheduledDate}</p>
              <p style="margin: 5px 0;"><strong>⏰ Scheduled Time:</strong> ${scheduledTime}</p>
              <p style="margin: 5px 0;"><strong>👨‍⚕️ Doctor:</strong> Dr. ${doctorName}</p>
            </div>

            <p><strong>⚡ Action Required:</strong></p>
            <p>Please log in to your patient dashboard and click the "Join Consultation" button to start your video consultation.</p>

            <p style="color: #dc2626; font-weight: bold;">⏱️ Please join as soon as possible. The doctor is waiting for you!</p>
          </div>

          <div class="footer">
            <p>This is an automated notification from MediConsult Zero Trust Secure Telehealth Platform.</p>
            <p>For support, please contact our help desk.</p>
          </div>
        </div>
      </body>
      </html>
    `
  };

  try {
    await transporter.sendMail(mailOptions);
    console.log(`✅ Consultation started email sent to ${patientEmail}`);
    return true;
  } catch (error) {
    console.error('❌ Error sending consultation started email:', error);
    throw error;
  }
};

// Send waiting alert to doctor
export const sendWaitingAlertEmail = async (doctorEmail, doctorName, patientName, appointmentNumber, scheduledDate, scheduledTime) => {
  const mailOptions = {
    from: {
      name: 'MediConsult - Zero Trust Secure Telehealth Platform',
      address: process.env.EMAIL_USER || 'cn3581743@gmail.com'
    },
    to: doctorEmail,
    subject: `⏰ Patient Waiting - ${appointmentNumber}`,
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
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            border-radius: 10px;
            padding: 30px;
            color: white;
          }
          .content {
            background: white;
            color: #333;
            border-radius: 8px;
            padding: 25px;
            margin-top: 20px;
          }
          .header {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
          }
          .info-box {
            background: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
          }
          .alert-box {
            background: #fef2f2;
            border: 2px solid #dc2626;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
            text-align: center;
          }
          .footer {
            margin-top: 20px;
            font-size: 12px;
            opacity: 0.9;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">⏰ Patient is Waiting!</div>
          <p>Urgent: Your patient is ready for consultation</p>

          <div class="content">
            <p><strong>Dear Dr. ${doctorName},</strong></p>

            <div class="alert-box">
              <p style="margin: 0; font-size: 18px; font-weight: bold; color: #dc2626;">
                🚨 ${patientName} is waiting for you to start the consultation!
              </p>
            </div>

            <div class="info-box">
              <p style="margin: 5px 0;"><strong>📋 Appointment Number:</strong> ${appointmentNumber}</p>
              <p style="margin: 5px 0;"><strong>📅 Scheduled Date:</strong> ${scheduledDate}</p>
              <p style="margin: 5px 0;"><strong>⏰ Scheduled Time:</strong> ${scheduledTime}</p>
              <p style="margin: 5px 0;"><strong>👤 Patient:</strong> ${patientName}</p>
            </div>

            <p><strong>⚡ Action Required:</strong></p>
            <p>Please log in to your doctor dashboard and click the "Start Consultation" button to begin the video consultation session.</p>

            <p style="color: #dc2626; font-weight: bold;">⏱️ Your patient is waiting. Please start the consultation as soon as possible!</p>
          </div>

          <div class="footer">
            <p>This is an automated notification from MediConsult Zero Trust Secure Telehealth Platform.</p>
            <p>For support, please contact our help desk.</p>
          </div>
        </div>
      </body>
      </html>
    `
  };

  try {
    await transporter.sendMail(mailOptions);
    console.log(`✅ Waiting alert email sent to ${doctorEmail}`);
    return true;
  } catch (error) {
    console.error('❌ Error sending waiting alert email:', error);
    throw error;
  }
};

// Send session terminated notification to patient (lockout max attempts)
export const sendLockoutTerminatedEmail = async (patientEmail, patientName, doctorName) => {
  const mailOptions = {
    from: {
      name: 'MediConsult - Zero Trust Secure Telehealth Platform',
      address: process.env.EMAIL_USER || 'cn3581743@gmail.com'
    },
    to: patientEmail,
    subject: '🔒 Consultation Ended — Security Verification Failed',
    html: `
      <!DOCTYPE html>
      <html>
      <head>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }
          .container { background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%); border-radius: 10px; padding: 30px; color: white; }
          .content { background: white; color: #333; border-radius: 8px; padding: 25px; margin-top: 20px; }
          .alert-box { background: #fef2f2; border: 2px solid #dc2626; padding: 15px; margin: 15px 0; border-radius: 8px; text-align: center; }
          .footer { margin-top: 20px; font-size: 12px; opacity: 0.9; }
        </style>
      </head>
      <body>
        <div class="container">
          <h2 style="margin-top: 0;">🔒 Consultation Session Ended</h2>
          <p>Security verification could not be completed</p>
          <div class="content">
            <p><strong>Dear ${patientName},</strong></p>
            <p>Your consultation with <strong>Dr. ${doctorName}</strong> has been <strong>automatically terminated</strong> due to a security incident.</p>
            <div class="alert-box">
              <p style="margin: 0; font-size: 16px; font-weight: bold; color: #dc2626;">
                ⚠️ The Zero Trust verification system detected that the doctor failed identity verification 5 times consecutively.
              </p>
            </div>
            <p>For your safety and security, the session was ended immediately.</p>
            <p><strong>What to do next:</strong></p>
            <ul>
              <li>Contact your doctor directly to reschedule the appointment.</li>
              <li>If you believe this is an error, please contact our support team.</li>
            </ul>
            <p style="color: #dc2626; font-weight: bold;">This action was taken automatically by the MediConsult Zero Trust Security System.</p>
          </div>
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
    await transporter.sendMail(mailOptions);
    console.log(`✅ Lockout termination email sent to ${patientEmail}`);
    return true;
  } catch (error) {
    console.error('❌ Error sending lockout termination email:', error);
    throw error;
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

