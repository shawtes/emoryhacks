import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Suspense } from 'react'
import ProtectedRoute from './components/ProtectedRoute'
import PageLoader from './components/PageLoader'
import {
  DoctorDashboardPage,
  DoctorLoginPage,
  DoctorSignupPage,
  HomePage,
  PatientAssessmentPage,
  PatientDashboardPage,
  PatientLoginPage,
  PatientSignupPage,
  TechStackPage,
} from './routes/lazyPages'
import { FirebaseProvider } from './context/FirebaseContext'
import { AuthProvider } from './context/AuthContext'

function App() {
  return (
    <FirebaseProvider>
      <AuthProvider>
        <BrowserRouter>
          <Suspense fallback={<PageLoader />}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/doctor/login" element={<DoctorLoginPage />} />
              <Route path="/doctor/signup" element={<DoctorSignupPage />} />
              <Route path="/patient/login" element={<PatientLoginPage />} />
              <Route path="/patient/signup" element={<PatientSignupPage />} />
              <Route path="/login" element={<Navigate to="/doctor/login" replace />} />
              <Route path="/signup" element={<Navigate to="/doctor/signup" replace />} />
              <Route
                path="/patientassessment"
                element={
                  <ProtectedRoute requiredRole="patient">
                    <PatientAssessmentPage />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/doctor/dashboard"
                element={
                  <ProtectedRoute requiredRole="doctor">
                    <DoctorDashboardPage />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/patient/dashboard"
                element={
                  <ProtectedRoute requiredRole="patient">
                    <PatientDashboardPage />
                  </ProtectedRoute>
                }
              />
              <Route path="/tech-stack" element={<TechStackPage />} />
              <Route path="/assessment" element={<Navigate to="/patientassessment" replace />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
        </BrowserRouter>
      </AuthProvider>
    </FirebaseProvider>
  )
}

export default App

