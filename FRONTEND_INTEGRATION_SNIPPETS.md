# Frontend Integration Code Snippets

## React/Axios Integration Examples

### 1. Patient - Create Appointment

```javascript
// In your PatientDashboard or BookAppointment component
import axios from 'axios';
import { useState } from 'react';
import toast from 'react-hot-toast';

const BookAppointment = ({ doctorId, onSuccess }) => {
  const [formData, setFormData] = useState({
    reason: '',
    preferredTime: 'Morning',
    preferredDates: 'Weekdays',
    additionalNotes: ''
  });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post('/api/appointments', {
        doctorId,
        ...formData
      });

      toast.success(`Appointment created! Number: ${response.data.data.appointment.appointmentNumber}`);
      onSuccess && onSuccess(response.data.data.appointment);
    } catch (error) {
      toast.error(error.response?.data?.message || 'Failed to create appointment');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <textarea
        placeholder="Reason for appointment"
        value={formData.reason}
        onChange={(e) => setFormData({...formData, reason: e.target.value})}
        required
      />
      
      <select
        value={formData.preferredTime}
        onChange={(e) => setFormData({...formData, preferredTime: e.target.value})}
      >
        <option value="Morning">Morning</option>
        <option value="Afternoon">Afternoon</option>
        <option value="Evening">Evening</option>
        <option value="Night">Night</option>
      </select>

      <select
        value={formData.preferredDates}
        onChange={(e) => setFormData({...formData, preferredDates: e.target.value})}
      >
        <option value="Weekdays">Weekdays</option>
        <option value="Weekends">Weekends</option>
        <option value="Any">Any</option>
      </select>

      <textarea
        placeholder="Additional notes (optional)"
        value={formData.additionalNotes}
        onChange={(e) => setFormData({...formData, additionalNotes: e.target.value})}
      />

      <button type="submit" disabled={loading}>
        {loading ? 'Submitting...' : 'Book Appointment'}
      </button>
    </form>
  );
};
```

### 2. Patient - View My Appointments

```javascript
import { useState, useEffect } from 'react';
import axios from 'axios';

const MyAppointments = () => {
  const [appointments, setAppointments] = useState([]);
  const [filter, setFilter] = useState('all'); // 'all', 'Pending', 'Approved', 'Rejected'
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAppointments();
  }, [filter]);

  const fetchAppointments = async () => {
    try {
      const url = filter === 'all' 
        ? '/api/appointments/my-appointments'
        : `/api/appointments/my-appointments?status=${filter}`;
      
      const response = await axios.get(url);
      setAppointments(response.data.data.appointments);
    } catch (error) {
      console.error('Error fetching appointments:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = async (appointmentId) => {
    if (!window.confirm('Are you sure you want to cancel this appointment?')) return;

    try {
      await axios.delete(`/api/appointments/${appointmentId}`);
      toast.success('Appointment cancelled successfully');
      fetchAppointments();
    } catch (error) {
      toast.error(error.response?.data?.message || 'Failed to cancel appointment');
    }
  };

  return (
    <div>
      {/* Filter Tabs */}
      <div className="tabs">
        <button onClick={() => setFilter('all')}>All</button>
        <button onClick={() => setFilter('Pending')}>Pending</button>
        <button onClick={() => setFilter('Approved')}>Approved</button>
        <button onClick={() => setFilter('Rejected')}>Rejected</button>
      </div>

      {/* Appointments List */}
      {appointments.map(apt => (
        <div key={apt._id} className={`appointment-card status-${apt.status.toLowerCase()}`}>
          <h3>{apt.appointmentNumber}</h3>
          <p>Doctor: Dr. {apt.doctorId.firstName} {apt.doctorId.lastName}</p>
          <p>Specialization: {apt.doctorId.specialization}</p>
          <p>Status: <span className={`badge-${apt.status.toLowerCase()}`}>{apt.status}</span></p>
          
          {apt.status === 'Approved' && (
            <>
              <p>Date: {new Date(apt.appointmentDate).toLocaleDateString()}</p>
              <p>Time: {apt.appointmentTimeFrom} - {apt.appointmentTimeTo}</p>
              {apt.doctorNote && <p>Note: {apt.doctorNote}</p>}
            </>
          )}
          
          {apt.status === 'Pending' && (
            <button onClick={() => handleCancel(apt._id)}>Cancel</button>
          )}
        </div>
      ))}
    </div>
  );
};
```

### 3. Doctor - View and Manage Appointments

```javascript
const DoctorAppointments = () => {
  const [appointments, setAppointments] = useState([]);
  const [filter, setFilter] = useState('Pending');
  const [selectedAppointment, setSelectedAppointment] = useState(null);

  useEffect(() => {
    fetchAppointments();
  }, [filter]);

  const fetchAppointments = async () => {
    try {
      const response = await axios.get(
        `/api/appointments/doctor-appointments?status=${filter}`
      );
      setAppointments(response.data.data.appointments);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      <div className="filter-tabs">
        <button onClick={() => setFilter('Pending')}>Pending</button>
        <button onClick={() => setFilter('Approved')}>Approved</button>
        <button onClick={() => setFilter('Rejected')}>Rejected</button>
      </div>

      {appointments.map(apt => (
        <div key={apt._id} className="appointment-card">
          <h3>{apt.appointmentNumber}</h3>
          <p>Patient: {apt.patientId.fullName}</p>
          <p>Age: {apt.patientId.age} | Gender: {apt.patientId.gender}</p>
          <p>Reason: {apt.reason}</p>
          <p>Preferred Time: {apt.preferredTime}</p>
          <p>Preferred Dates: {apt.preferredDates}</p>
          
          {apt.status === 'Pending' && (
            <div>
              <button onClick={() => setSelectedAppointment({...apt, action: 'approve'})}>
                Approve
              </button>
              <button onClick={() => setSelectedAppointment({...apt, action: 'reject'})}>
                Reject
              </button>
            </div>
          )}
        </div>
      ))}

      {selectedAppointment && (
        <AppointmentActionModal
          appointment={selectedAppointment}
          onClose={() => setSelectedAppointment(null)}
          onSuccess={fetchAppointments}
        />
      )}
    </div>
  );
};
```

### 4. Doctor - Approve Appointment Modal

```javascript
const ApproveAppointmentModal = ({ appointment, onClose, onSuccess }) => {
  const [formData, setFormData] = useState({
    appointmentDate: '',
    appointmentTimeFrom: '',
    appointmentTimeTo: '',
    doctorNote: ''
  });

  const handleApprove = async (e) => {
    e.preventDefault();

    try {
      await axios.put(`/api/appointments/${appointment._id}/approve`, formData);
      toast.success('Appointment approved! Patient will receive an email.');
      onSuccess();
      onClose();
    } catch (error) {
      toast.error(error.response?.data?.message || 'Failed to approve');
    }
  };

  return (
    <div className="modal">
      <h2>Approve Appointment</h2>
      <p>Appointment: {appointment.appointmentNumber}</p>
      <p>Patient: {appointment.patientId.fullName}</p>

      <form onSubmit={handleApprove}>
        <input
          type="date"
          value={formData.appointmentDate}
          onChange={(e) => setFormData({...formData, appointmentDate: e.target.value})}
          required
        />

        <input
          type="time"
          placeholder="From"
          value={formData.appointmentTimeFrom}
          onChange={(e) => setFormData({...formData, appointmentTimeFrom: e.target.value})}
          required
        />

        <input
          type="time"
          placeholder="To"
          value={formData.appointmentTimeTo}
          onChange={(e) => setFormData({...formData, appointmentTimeTo: e.target.value})}
          required
        />

        <textarea
          placeholder="Doctor's note (optional)"
          value={formData.doctorNote}
          onChange={(e) => setFormData({...formData, doctorNote: e.target.value})}
        />

        <button type="submit">Approve Appointment</button>
        <button type="button" onClick={onClose}>Cancel</button>
      </form>
    </div>
  );
};
```

### 5. Doctor - Reject Appointment Modal

```javascript
const RejectAppointmentModal = ({ appointment, onClose, onSuccess }) => {
  const [doctorNote, setDoctorNote] = useState('');

  const handleReject = async () => {
    try {
      await axios.put(`/api/appointments/${appointment._id}/reject`, { doctorNote });
      toast.success('Appointment rejected. Patient will receive an email.');
      onSuccess();
      onClose();
    } catch (error) {
      toast.error(error.response?.data?.message || 'Failed to reject');
    }
  };

  return (
    <div className="modal">
      <h2>Reject Appointment</h2>
      <p>Appointment: {appointment.appointmentNumber}</p>
      <p>Patient: {appointment.patientId.fullName}</p>

      <textarea
        placeholder="Reason for rejection (optional)"
        value={doctorNote}
        onChange={(e) => setDoctorNote(e.target.value)}
      />

      <button onClick={handleReject}>Reject Appointment</button>
      <button onClick={onClose}>Cancel</button>
    </div>
  );
};
```

---

## CSS Styling Suggestions

```css
/* Status Badges */
.badge-pending {
  background: #fbbf24;
  color: #78350f;
  padding: 4px 12px;
  border-radius: 12px;
  font-weight: 600;
}

.badge-approved {
  background: #10b981;
  color: white;
  padding: 4px 12px;
  border-radius: 12px;
  font-weight: 600;
}

.badge-rejected {
  background: #ef4444;
  color: white;
  padding: 4px 12px;
  border-radius: 12px;
  font-weight: 600;
}

/* Appointment Cards */
.appointment-card {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
}

.appointment-card.status-pending {
  border-left: 4px solid #fbbf24;
}

.appointment-card.status-approved {
  border-left: 4px solid #10b981;
}

.appointment-card.status-rejected {
  border-left: 4px solid #ef4444;
}
```

---

**Ready to integrate! 🚀**

