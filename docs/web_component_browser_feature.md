# FastAPI Web Component Browser Feature

## 📋 Feature Overview

**Feature Name:** Interactive ICA Component Classification Browser  
**Type:** New Feature  
**Priority:** High  
**Estimated Effort:** 2-3 days for MVP  

### 🎯 Business Requirements

Create a FastAPI web interface that allows users to:
1. **Browse** composite IC images generated from MNE raw and ICA files
2. **Review** existing CSV classification results with visual component display
3. **Override** classifications interactively through a web UI
4. **Export** override results as CSV files for pipeline integration
5. **Process** entire folders containing raw, ICA, and results files

### 🔧 Technical Specifications

#### Input Requirements
- **Folder Structure:**
  ```
  /input/folder/
  ├── subject_001.set                    # Raw EEG data
  ├── subject_001.fif                    # ICA decomposition  
  ├── subject_001_icvis_results.csv      # Existing results (optional)
  ├── subject_002.set                    # Additional subjects...
  └── subject_002_icvis_results.csv      # Additional results...
  ```

#### Output Requirements
- **Override CSV Files:**
  ```
  /input/folder/
  ├── subject_001_icvis_results_override.csv
  └── subject_002_icvis_results_override.csv
  ```

#### File Naming Convention
- Input files follow existing ICVision basename extraction
- Override files use `{basename}_icvis_results_override.csv` suffix
- Compatible with existing pipeline integration

## 🏗️ Technical Architecture

### Code Reuse Analysis

**Existing Components to Leverage:**
- `src/icvision/plotting.py`: Component image generation (✅ Ready)
- `src/icvision/core.py`: Core workflow functions (✅ Ready)
- `src/icvision/utils.py`: File loading/saving utilities (✅ Ready)
- CSV schema and validation (✅ Ready)

**New Components Required:**
```
src/icvision/web/
├── __init__.py              # Web module initialization
├── app.py                   # FastAPI application
├── models.py                # Pydantic models for API contracts
├── routes.py                # API endpoint definitions
├── templates/               # HTML templates
│   └── component_browser.html
└── static/                  # CSS/JS assets
    ├── style.css
    └── app.js
```

### API Endpoints Design

#### 1. Folder Processing Endpoint
```python
@app.post("/process-folder")
async def process_folder(
    folder_path: str,
    confidence_threshold: float = 0.8
) -> ProcessFolderResponse:
    """
    Scan folder for EEG files, load data, generate component images.
    Returns session ID for subsequent operations.
    """
```

#### 2. Component Data Retrieval
```python
@app.get("/components/{session_id}")
async def get_components(session_id: str) -> ComponentDataResponse:
    """
    Retrieve component data with base64-encoded images and existing classifications.
    """
```

#### 3. Classification Override
```python
@app.post("/override-classification")
async def override_classification(
    session_id: str,
    component_id: str,
    new_label: str,
    confidence: float,
    reason: str
) -> OverrideResponse:
    """
    Save user's classification override for specific component.
    """
```

#### 4. Export Override Results
```python
@app.get("/export-overrides/{session_id}")
async def export_overrides(session_id: str) -> FileResponse:
    """
    Generate and download override CSV files for pipeline integration.
    """
```

### Data Models

#### Component Data Model
```python
class ComponentData(BaseModel):
    component_index: int
    component_name: str
    current_label: str
    confidence: float
    reason: str
    exclude_vision: bool
    image_base64: str
    override_label: Optional[str] = None
    override_confidence: Optional[float] = None
    override_reason: Optional[str] = None
```

#### Session Data Model
```python
class SessionData(BaseModel):
    session_id: str
    folder_path: str
    subjects: List[SubjectData]
    created_at: datetime
    last_modified: datetime
```

## 🔄 Integration Strategy

### Pipeline Integration Points

1. **Override Detection in Core Pipeline:**
   ```python
   # In core.py label_components():
   override_file = f"{basename}_icvis_results_override.csv"
   if Path(override_file).exists():
       results_df = pd.read_csv(override_file)
       # Skip OpenAI API calls, use override results
       return process_override_results(results_df)
   ```

2. **Existing CSV Schema Compatibility:**
   ```python
   # Override CSV maintains same schema as existing results:
   {
       "component_index": int,
       "component_name": str,
       "label": str,
       "confidence": float,
       "reason": str,
       "exclude_vision": bool
   }
   ```

### Memory Management Strategy

```python
# Async processing for large datasets:
@app.post("/process-folder")
async def process_folder(folder_path: str):
    return await asyncio.to_thread(process_folder_sync, folder_path)

# Image conversion helper:
def fig_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)  # Important: cleanup matplotlib figures
    return img_base64
```

## 📦 Implementation Plan

### Phase 1: Core Infrastructure (Day 1)

**Tasks:**
1. Add FastAPI dependencies to `pyproject.toml`
   ```toml
   [project.optional-dependencies]
   web = [
       "fastapi>=0.100.0",
       "uvicorn>=0.20.0", 
       "jinja2>=3.1.0",
       "python-multipart>=0.0.6"
   ]
   ```

2. Create web module structure
3. Implement basic FastAPI app with folder processing endpoint
4. Add session management with in-memory storage

**Deliverables:**
- Basic web server that can scan folders and identify EEG files
- Session-based data storage
- Initial API endpoint structure

### Phase 2: Component Processing (Day 2)

**Tasks:**
1. Integrate existing plotting functions for web use
2. Implement component image generation with base64 encoding
3. Add CSV results loading and parsing
4. Create component data API endpoints

**Key Integration Points:**
```python
# Modify plotting.py to add web helper:
def plot_component_for_web(ica_obj, raw_obj, component_idx, psd_fmax=None):
    """Generate component plot and return as base64 string."""
    fig = plot_component_for_classification(
        ica_obj, raw_obj, component_idx, 
        output_dir=None, return_fig_object=True, psd_fmax=psd_fmax
    )
    return fig_to_base64(fig)
```

**Deliverables:**
- Component image generation for web
- API endpoints returning component data with images
- CSV results integration

### Phase 3: Web Interface (Day 3)

**Tasks:**
1. Create HTML template for component browser
2. Implement JavaScript for interactive classification
3. Add override functionality and export features
4. Implement proper error handling and validation

**UI Components:**
- Component grid view with images
- Classification dropdown with confidence slider
- Bulk operations (select all artifacts, etc.)
- Export button for override CSV generation

**Deliverables:**
- Complete web interface
- Override classification functionality
- CSV export capability

### Phase 4: Testing & Documentation (Day 3 continued)

**Tasks:**
1. Add comprehensive tests for web endpoints
2. Test with sample EEG datasets
3. Update documentation and CLI help
4. Performance testing with large datasets

**Test Coverage:**
- Unit tests for new web utilities
- Integration tests for API endpoints
- End-to-end tests with sample data
- Performance tests for memory usage

## 🔒 Security Considerations

### File Access Security
```python
# Path traversal protection:
def validate_folder_path(folder_path: str) -> Path:
    path = Path(folder_path).resolve()
    if not path.exists() or not path.is_dir():
        raise HTTPException(400, "Invalid folder path")
    return path

# File type validation:
ALLOWED_EXTENSIONS = {'.set', '.fif', '.edf', '.bdf', '.vhdr', '.csv'}
def validate_file_extension(file_path: Path) -> bool:
    return file_path.suffix.lower() in ALLOWED_EXTENSIONS
```

### Rate Limiting
```python
# Add rate limiting for API endpoints:
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/process-folder")
@limiter.limit("5/minute")  # Limit folder processing requests
async def process_folder(request: Request, folder_path: str):
    # Implementation
```

## 📊 Success Metrics

### Functional Requirements
- ✅ Successfully load and display component images from folder input
- ✅ Pre-populate classifications from existing CSV results
- ✅ Allow interactive override of classifications
- ✅ Generate override CSV files compatible with existing pipeline
- ✅ Handle multiple subjects in single folder

### Performance Requirements
- Process folders with 10+ subjects within 30 seconds
- Handle component images up to 50 components per subject
- Memory usage remains stable during long sessions
- Responsive UI with <2 second image loading

### Quality Requirements
- 100% test coverage for new web module
- Zero breaking changes to existing CLI functionality
- Comprehensive error handling and user feedback
- Cross-browser compatibility (Chrome, Firefox, Safari)

## 🚀 Deployment Instructions

### Development Setup
```bash
# Install with web dependencies
pip install -e ".[dev,test,web]"

# Start development server
uvicorn icvision.web.app:app --reload --port 8000

# Access web interface
open http://localhost:8000
```

### Production Deployment
```bash
# Install production dependencies
pip install -e ".[web]"

# Start production server
uvicorn icvision.web.app:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📚 Additional Resources

### Related Documentation
- [ICVision Core API](../api/core.rst)
- [Plotting Module Documentation](../api/plotting.rst)
- [CSV Results Schema](../api/reports.rst)

### Example Usage
```python
# Example folder structure for testing:
test_data/
├── subject_001.set
├── subject_001.fif  
├── subject_001_icvis_results.csv
├── subject_002.set
└── subject_002.fif

# Expected output:
test_data/
├── subject_001_icvis_results_override.csv
└── subject_002_icvis_results_override.csv
```

---

## 📝 Implementation Progress Log

### ✅ **Phase 1: Core Infrastructure (COMPLETED)**
**Date:** July 18, 2025  
**Developer Team:** Collaborative pair programming (Dev1 + Dev2 + PM)

**Completed Tasks:**
- ✅ Added FastAPI dependencies to `pyproject.toml` (web optional dependency)
- ✅ Created complete web module structure (`src/icvision/web/`)
- ✅ Implemented FastAPI app with session management
- ✅ Added comprehensive Pydantic models for type safety
- ✅ Created API endpoint structure with proper error handling

**Key Decisions Made:**
- Removed slowapi dependency for MVP (rate limiting can be added later)
- Used in-memory session storage (Redis recommended for production)
- Implemented graceful import handling for optional web dependencies

### ✅ **Phase 3: Web Interface (COMPLETED)**
**Date:** July 18, 2025

**Completed Tasks:**
- ✅ Created responsive HTML template with modern CSS
- ✅ Implemented JavaScript for interactive component classification
- ✅ Added real-time override functionality with confidence scoring
- ✅ Included export capabilities for CSV generation
- ✅ Added loading states and error handling in UI

**UI Features Implemented:**
- Folder path input with validation
- Component grid display with images
- Interactive classification dropdowns
- Confidence score adjustment
- Export button for override CSV files
- Status messages and loading indicators

### ⚠️ **Phase 2: Component Processing (PARTIAL)**
**Status:** Integration issues identified

**Completed:**
- ✅ Created web utilities for file scanning and validation
- ✅ Implemented base64 image conversion helpers
- ✅ Added CSV export functionality for override results

**Issues to Resolve:**
- 🔧 Plotting function integration with existing codebase
- 🔧 Relative import resolution in web utils
- 🔧 Type compatibility between web and core modules

### 📋 **Next Development Iteration**

**Priority 1: Complete Phase 2**
1. Fix plotting function integration issues
2. Resolve import dependencies between web and core modules
3. Add proper error handling for edge cases
4. Test with sample EEG data

**Priority 2: Testing & Polish**
1. Add comprehensive unit tests for web endpoints
2. Integration testing with real EEG files
3. Performance testing with large datasets
4. Cross-browser compatibility testing

**Priority 3: Production Readiness**
1. Add rate limiting (slowapi integration)
2. Implement Redis session storage
3. Add authentication/authorization
4. Docker containerization

---

**Implementation Status:** Phase 1 & 3 Complete, Phase 2 Partial  
**Current Commit:** c8e3928 - "Implement Phase 1-3 of FastAPI web component browser"  
**Next Milestone:** Complete Phase 2 integration  
**Review Required:** Senior Developer review of integration approach