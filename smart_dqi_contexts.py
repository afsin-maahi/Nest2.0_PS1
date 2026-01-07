"""
Context-aware weight adjustments for Smart DQI
Based on therapeutic area, trial phase, and timeline
"""

# Base weights (same as Basic DQI)
BASE_WEIGHTS = {
    'visits': 0.25,
    'pages': 0.20,
    'queries': 0.20,
    'accuracy': 0.15,
    'verification': 0.10,
    'safety': 0.10
}

# Therapeutic Area Multipliers
THERAPEUTIC_AREAS = {
    'oncology': {
        'description': 'Cancer trials - highest safety priority',
        'multipliers': {
            'safety': 4.0,      # 4x weight - CRITICAL
            'visits': 1.2,      # Tumor assessments important
            'accuracy': 1.3,    # Dosing precision critical
            'pages': 0.8,       # Can catch up later
            'queries': 1.0,
            'verification': 1.0
        }
    },
    'cardiology': {
        'description': 'Cardiovascular trials',
        'multipliers': {
            'safety': 2.0,      # 2x weight
            'visits': 1.5,      # Regular monitoring crucial
            'accuracy': 1.2,
            'pages': 0.9,
            'queries': 1.1,
            'verification': 1.0
        }
    },
    'standard': {
        'description': 'Standard trials',
        'multipliers': {
            'safety': 1.0,
            'visits': 1.0,
            'pages': 1.0,
            'queries': 1.0,
            'accuracy': 1.0,
            'verification': 1.0
        }
    }
}

# Trial Phase Multipliers
TRIAL_PHASES = {
    'phase3': {
        'description': 'Phase 3 - Submission ready',
        'multipliers': {
            'pages': 1.4,       # Everything must be complete
            'verification': 1.3,
            'queries': 1.3,
            'visits': 1.2,
            'accuracy': 1.2,
            'safety': 1.5
        }
    },
    'phase2': {
        'description': 'Phase 2 - Dose finding',
        'multipliers': {
            'accuracy': 1.3,
            'visits': 1.2,
            'safety': 1.5,
            'pages': 1.0,
            'queries': 1.1,
            'verification': 1.0
        }
    }
}

# Timeline Multipliers
TIMELINES = {
    'late': {
        'description': 'Late enrollment - database lock approaching',
        'multipliers': {
            'pages': 1.5,       # MUST complete
            'queries': 1.5,     # Close all queries
            'verification': 1.4,
            'visits': 1.3,
            'accuracy': 1.2,
            'safety': 1.3
        }
    },
    'mid': {
        'description': 'Mid enrollment - standard expectations',
        'multipliers': {
            'safety': 1.0,
            'visits': 1.0,
            'pages': 1.0,
            'queries': 1.0,
            'accuracy': 1.0,
            'verification': 1.0
        }
    }
}

def get_smart_weights(therapeutic_area='standard', phase='phase3', timeline='late'):
    """
    Calculate context-aware weights
    """
    # Start with base weights
    weights = BASE_WEIGHTS.copy()
    
    # Apply therapeutic area multipliers
    area_mult = THERAPEUTIC_AREAS[therapeutic_area]['multipliers']
    for key in weights:
        weights[key] *= area_mult[key]
    
    # Apply phase multipliers
    phase_mult = TRIAL_PHASES[phase]['multipliers']
    for key in weights:
        weights[key] *= phase_mult[key]
    
    # Apply timeline multipliers
    timeline_mult = TIMELINES[timeline]['multipliers']
    for key in weights:
        weights[key] *= timeline_mult[key]
    
    # Normalize to sum to 1.0
    total = sum(weights.values())
    normalized = {k: v/total for k, v in weights.items()}
    
    return normalized

# Test it
if __name__ == "__main__":
    print("🧪 SMART DQI WEIGHT EXAMPLES:\n")
    
    scenarios = [
        ('Standard Trial', 'standard', 'phase3', 'mid'),
        ('Oncology Phase 3 (Late)', 'oncology', 'phase3', 'late'),
        ('Cardiology Phase 2', 'cardiology', 'phase2', 'mid')
    ]
    
    for name, area, phase, timeline in scenarios:
        weights = get_smart_weights(area, phase, timeline)
        print(f"📋 {name}:")
        for component, weight in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"   {component:15s}: {weight:6.1%}")
        print()
