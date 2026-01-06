"""
Tests for VDG package.
"""

import pytest
import numpy as np


class TestFeatureTracker:
    """Tests for FeatureTracker class."""
    
    def test_import(self):
        """Test that FeatureTracker can be imported."""
        from vdg.tracking import FeatureTracker
        assert FeatureTracker is not None
    
    def test_initialization(self):
        """Test FeatureTracker initialization."""
        from vdg.tracking import FeatureTracker
        
        tracker = FeatureTracker(num_features=50)
        assert tracker.num_features == 50
        assert tracker.points is None
        assert tracker.frame_count == 0
    
    def test_with_roi(self):
        """Test FeatureTracker with ROI."""
        from vdg.tracking import FeatureTracker
        
        roi = (100, 100, 200, 200)
        tracker = FeatureTracker(num_features=20, initial_roi=roi)
        assert tracker.initial_roi == roi
        assert tracker.use_dynamic_roi is True


class TestOutputSpec:
    """Tests for OutputSpec parser."""
    
    def test_simple_spec(self):
        """Test parsing simple spec."""
        from vdg.outputs import OutputSpec
        
        spec = OutputSpec("previewtrack")
        assert spec.output_type == "previewtrack"
        assert len(spec.options) == 0
    
    def test_spec_with_options(self):
        """Test parsing spec with options."""
        from vdg.outputs import OutputSpec
        
        spec = OutputSpec("previewtrack=filename=test.mp4:mode=stab")
        assert spec.output_type == "previewtrack"
        assert spec.get('filename') == "test.mp4"
        assert spec.get('mode') == "stab"
    
    def test_get_with_default(self):
        """Test getting option with default."""
        from vdg.outputs import OutputSpec
        
        spec = OutputSpec("csv")
        assert spec.get('missing', 'default') == 'default'
    
    def test_get_int(self):
        """Test getting integer option."""
        from vdg.outputs import OutputSpec
        
        spec = OutputSpec("trackers=type=2")
        assert spec.get_int('type') == 2
        assert spec.get_int('missing', 1) == 1
    
    def test_get_bool(self):
        """Test getting boolean option."""
        from vdg.outputs import OutputSpec
        
        spec = OutputSpec("previewtrack=showids=true")
        assert spec.get_bool('showids') is True
        assert spec.get_bool('missing') is False


class TestOutputManager:
    """Tests for OutputManager."""
    
    def test_add_output(self):
        """Test adding outputs."""
        from vdg.outputs import OutputManager
        
        manager = OutputManager("input.mp4")
        manager.add_output("previewtrack")
        manager.add_output("csv")
        
        assert len(manager) == 2
    
    def test_invalid_output_type(self):
        """Test that invalid output type raises error."""
        from vdg.outputs import OutputManager
        
        manager = OutputManager("input.mp4")
        with pytest.raises(ValueError):
            manager.add_output("invalid_type")


class TestTrackIO:
    """Tests for track I/O utilities."""
    
    def test_parse_crv_line(self):
        """Test parsing CRV format line."""
        from vdg.tracking.track_io import parse_track_line
        
        result = parse_track_line("100 [[ 0.5, 0.3]]")
        assert result == (100, 0.5, 0.3)
    
    def test_parse_simple_line(self):
        """Test parsing simple format line."""
        from vdg.tracking.track_io import parse_track_line
        
        result = parse_track_line("100 0.5 0.3")
        assert result == (100, 0.5, 0.3)
    
    def test_parse_empty_line(self):
        """Test parsing empty line returns None."""
        from vdg.tracking.track_io import parse_track_line
        
        result = parse_track_line("")
        assert result is None


class TestVideoProperties:
    """Tests for VideoProperties."""
    
    def test_is_portrait(self):
        """Test portrait detection."""
        from vdg.core.video import VideoProperties
        
        portrait = VideoProperties(1080, 1920, 30, 100)
        landscape = VideoProperties(1920, 1080, 30, 100)
        
        assert portrait.is_portrait is True
        assert landscape.is_portrait is False
    
    def test_to_dict(self):
        """Test conversion to dict."""
        from vdg.core.video import VideoProperties
        
        props = VideoProperties(1920, 1080, 30, 100)
        d = props.to_dict()
        
        assert d['width'] == 1920
        assert d['height'] == 1080
        assert d['fps'] == 30
        assert d['frame_count'] == 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
