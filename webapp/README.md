# Frontend Architecture

## Overview

This is a React/TypeScript web application for dementia detection through speech/audio analysis. Built with modern web technologies for scalability and maintainability.

## Tech Stack

- **Framework**: React 18.2
- **Language**: TypeScript 5.0
- **Build Tool**: Vite 5.4
- **Routing**: React Router DOM 6.20
- **HTTP Client**: Axios 1.6
- **Linting**: ESLint 9 with TypeScript ESLint 8
- **Styling**: CSS (Single stylesheet approach)

## Project Structure

```
webapp/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── AudioRecorder.tsx    # Audio recording component
│   │   ├── FileUploader.tsx     # File upload with drag-and-drop
│   │   ├── PatientForm.tsx      # Patient information form
│   │   ├── ProtectedRoute.tsx   # Route protection wrapper
│   │   └── ResultsDisplay.tsx   # Results visualization
│   ├── pages/               # Page components
│   │   ├── Home.tsx            # Landing page
│   │   ├── Login.tsx           # Login page (UI only)
│   │   ├── Signup.tsx          # Signup page (UI only)
│   │   └── Assessment.tsx      # Main assessment workflow
│   ├── App.tsx              # Root component with routing
│   ├── main.tsx             # Application entry point
│   ├── types.ts             # TypeScript type definitions
│   ├── styles.css           # Single stylesheet (all styles)
│   └── index.css            # Global resets + imports styles.css
├── public/                  # Static assets
├── Dockerfile               # Production Docker image
├── nginx.conf               # Nginx configuration
├── vite.config.ts           # Vite build configuration
├── tsconfig.json            # TypeScript configuration
├── package.json             # Dependencies and scripts
└── eslint.config.js         # ESLint configuration (ESM)

```

## Architecture Patterns

### Component-Based Architecture

- **Pages**: Top-level route components (`Home`, `Login`, `Signup`, `Assessment`)
- **Components**: Reusable UI components (`AudioRecorder`, `FileUploader`, etc.)
- **Single Responsibility**: Each component has a focused purpose

### Routing

Uses React Router DOM for client-side routing:

```typescript
/                    → Home page
/login               → Login page (UI only, no auth)
/signup              → Signup page (UI only, no auth)
/assessment          → Main assessment workflow (protected route)
```

### State Management

- **Local State**: React hooks (`useState`, `useEffect`) for component state
- **No Global State**: Currently no Redux/Zustand (can be added if needed)
- **Props Drilling**: Data passed through component props

### Styling Architecture

**Single Stylesheet Approach** - All styles in `styles.css`:

- CSS Variables for theming (colors, spacing, etc.)
- Reusable component classes (`.btn-primary`, `.card`, `.form-group`)
- Page-specific styles organized by section
- Component-specific styles with descriptive class names
- No CSS modules or styled-components (keeps it simple)

See `CSS_ARCHITECTURE.md` for detailed styling guidelines.

## Key Components

### Assessment Page (`pages/Assessment.tsx`)

Main workflow component managing the multi-step assessment:

1. **Audio Step** (default): Record or upload audio
2. **Results Step**: Display prediction results
3. **Patient Info Step** (optional): Edit patient information

**State Management**:
- `currentStep`: Current step in workflow
- `patientInfo`: Patient demographic data
- `result`: ML prediction results
- `sessions`: Assessment history

### Audio Recorder (`components/AudioRecorder.tsx`)

Handles browser audio recording:
- Uses `MediaRecorder` API
- Records audio from microphone
- Provides preview and reset functionality
- Converts to File for API submission

### File Uploader (`components/FileUploader.tsx`)

File upload component:
- Drag-and-drop support
- File validation (audio formats)
- File preview with metadata
- Remove file functionality

### Results Display (`components/ResultsDisplay.tsx`)

Visualizes ML prediction results:
- Prediction probability display
- Progress bar visualization
- Confidence level badges
- Patient information display
- Clinical disclaimer

## API Integration

### Configuration

API URL configured via environment variable:
```typescript
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
```

### Endpoints

- `POST /predict` - Submit audio file for analysis
  - Body: `FormData` with audio file
  - Response: `PredictionResult` object

### Error Handling

- Network errors caught and displayed to user
- Loading states during API calls
- Error messages shown in UI

## Development

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

### Development Server

- Runs on `http://localhost:3000`
- Hot Module Replacement (HMR) enabled
- API proxy configured for `/api` → `http://localhost:8000`

### Environment Variables

Create `.env` file:
```
VITE_API_URL=http://localhost:8000
```

## Build Process

### Vite Build

1. **TypeScript Compilation**: `tsc` checks types
2. **Vite Build**: Bundles and optimizes assets
3. **Output**: `dist/` directory with static files

### Production Build

- Code splitting and tree-shaking
- Minification and compression
- Source maps generated
- Optimized asset names with hashes

## Code Quality

### TypeScript

- Strict type checking enabled
- Type definitions in `types.ts`
- Component props typed
- API responses typed

### ESLint

- ESLint 9 with flat config
- TypeScript ESLint plugin
- React hooks linting
- React refresh plugin

### Code Style

- Consistent component structure
- Descriptive variable names
- Comments for complex logic
- Error handling in async operations

## Browser Support

- Modern browsers (Chrome, Firefox, Safari, Edge)
- ES2020+ features
- MediaRecorder API (for audio recording)
- File API (for file uploads)

## Performance Considerations

### Optimization

- Code splitting by route
- Lazy loading (can be added for routes)
- Asset optimization (Vite handles automatically)
- CSS minification
- Gzip compression (via Nginx)

### Bundle Size

- Tree-shaking removes unused code
- Dynamic imports for large dependencies
- Optimized React production build

## Security

### Current Implementation

- No authentication (UI only)
- CORS handled by backend
- Input validation on forms
- XSS protection (React escapes by default)

### Future Enhancements

- JWT token storage
- Secure HTTP-only cookies
- CSRF protection
- Content Security Policy headers

## Testing

### Current Status

- No test suite (can be added)
- Manual testing recommended

### Recommended Testing Stack

- **Unit Tests**: Vitest + React Testing Library
- **E2E Tests**: Playwright or Cypress
- **Component Tests**: React Testing Library

## Accessibility

### Current Features

- Semantic HTML elements
- Form labels and inputs
- Button roles and states
- Keyboard navigation support

### Future Improvements

- ARIA labels for complex components
- Screen reader optimization
- Focus management
- High contrast mode support

## Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Check `VITE_API_URL` environment variable
   - Verify backend is running
   - Check CORS configuration

2. **Audio Recording Not Working**
   - Check browser permissions
   - Verify HTTPS (required for MediaRecorder)
   - Test microphone access

3. **Build Errors**
   - Clear `node_modules` and reinstall
   - Check TypeScript errors: `npm run build`
   - Verify Node.js version compatibility

4. **Styling Issues**
   - Check `styles.css` imports
   - Verify CSS variables are defined
   - Check browser DevTools for conflicts

## Contributing

### Code Style

- Use TypeScript for all new files
- Follow existing component patterns
- Add types for new data structures
- Update `styles.css` for new styles (don't create new CSS files)

### Adding Features

1. Create component in appropriate directory
2. Add types to `types.ts` if needed
3. Update routing in `App.tsx` if new page
4. Add styles to `styles.css` (no new CSS files)
5. Test in development mode
6. Update documentation

## Resources

- [React Documentation](https://react.dev)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Vite Guide](https://vitejs.dev/guide/)
- [React Router Docs](https://reactrouter.com/)
- [CSS Architecture Guide](./CSS_ARCHITECTURE.md)


