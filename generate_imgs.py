import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime, timedelta
import json

class HeartMonitorGenerator:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.colors = {
            'bg': '#000000',
            'frame': '#2c2c2c',
            'text': '#00ff00',
            'ecg': '#00ff00',
            'warning': '#ffff00',
            'alarm': '#ff0000',
            'blue': '#0080ff',
            'white': '#ffffff'
        }
        
    def generate_ecg_waveform(self, length=300, heart_rate=75):
        """Generate realistic ECG waveform"""
        # Calculate samples per beat based on heart rate
        samples_per_beat = int(60 * 4 / heart_rate)  # 4 samples per second
        
        # Generate multiple heartbeats
        ecg_data = []
        for i in range(length):
            beat_position = i % samples_per_beat
            beat_progress = beat_position / samples_per_beat
            
            # Generate QRS complex
            if 0.1 <= beat_progress <= 0.3:
                if 0.15 <= beat_progress <= 0.25:
                    # R wave (main spike)
                    amplitude = 0.8 * np.sin((beat_progress - 0.15) * np.pi / 0.1)
                else:
                    # Q and S waves
                    amplitude = -0.2 * np.sin((beat_progress - 0.1) * np.pi / 0.2)
            elif 0.35 <= beat_progress <= 0.6:
                # T wave
                amplitude = 0.3 * np.sin((beat_progress - 0.35) * np.pi / 0.25)
            else:
                # Baseline with small noise
                amplitude = random.uniform(-0.05, 0.05)
            
            ecg_data.append(amplitude + random.uniform(-0.02, 0.02))
        
        return np.array(ecg_data)
    
    def generate_vital_signs(self):
        """Generate realistic vital signs"""
        return {
            'heart_rate': random.randint(60, 120),
            'systolic_bp': random.randint(90, 160),
            'diastolic_bp': random.randint(60, 100),
            'spo2': random.randint(95, 100),
            'temperature': round(random.uniform(36.0, 39.0), 1),
            'respiratory_rate': random.randint(12, 25),
            'pulse_rate': random.randint(60, 120)
        }
    
    def create_monitor_display(self, output_path, patient_info=None):
        """Create a complete heart monitor display"""
        fig, ax = plt.subplots(figsize=(self.width/100, self.height/100), dpi=100)
        fig.patch.set_facecolor(self.colors['bg'])
        ax.set_facecolor(self.colors['bg'])
        
        # Remove axes
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')
        
        # Generate vital signs
        vitals = self.generate_vital_signs()
        
        # Header with patient info and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        patient_id = patient_info.get('id', f"RN:{random.randint(10000000, 99999999)}") if patient_info else f"RN:{random.randint(10000000, 99999999)}"
        
        # Draw header
        ax.text(2, 95, patient_id, fontsize=12, color=self.colors['text'], fontweight='bold')
        ax.text(70, 95, current_time, fontsize=12, color=self.colors['text'], fontweight='bold')
        
        # Draw ECG sections
        ecg_sections = ['ECG I', 'ECG II', 'ECG III']
        for i, section in enumerate(ecg_sections):
            y_base = 85 - i * 25
            
            # Section label
            ax.text(2, y_base, section, fontsize=10, color=self.colors['text'], fontweight='bold')
            
            # Generate and plot ECG waveform
            ecg_data = self.generate_ecg_waveform(300, vitals['heart_rate'])
            x_vals = np.linspace(10, 90, len(ecg_data))
            y_vals = y_base - 10 + ecg_data * 8
            ax.plot(x_vals, y_vals, color=self.colors['ecg'], linewidth=1.5)
            
            # Grid lines
            for x in range(10, 91, 5):
                ax.axvline(x, color='#003300', alpha=0.3, linewidth=0.5)
            for y in range(int(y_base-15), int(y_base-5), 2):
                ax.axhline(y, color='#003300', alpha=0.3, linewidth=0.5, xmin=0.1, xmax=0.9)
        
        # Vital signs display panel
        vitals_x = 75
        vitals_y = 70
        
        # Heart Rate
        ax.text(vitals_x, vitals_y, "HR", fontsize=10, color=self.colors['text'])
        ax.text(vitals_x + 5, vitals_y, "bpm", fontsize=8, color=self.colors['text'])
        ax.text(vitals_x + 12, vitals_y, str(vitals['heart_rate']), fontsize=16, color=self.colors['text'], fontweight='bold')
        
        # Blood Pressure
        ax.text(vitals_x, vitals_y - 8, "NIBP", fontsize=10, color=self.colors['text'])
        ax.text(vitals_x + 8, vitals_y - 8, "mmHg", fontsize=8, color=self.colors['text'])
        bp_text = f"{vitals['systolic_bp']}/{vitals['diastolic_bp']}"
        ax.text(vitals_x + 5, vitals_y - 12, bp_text, fontsize=14, color=self.colors['text'], fontweight='bold')
        ax.text(vitals_x + 5, vitals_y - 16, f"({int((vitals['systolic_bp'] + 2*vitals['diastolic_bp'])/3)})", fontsize=10, color=self.colors['text'])
        
        # SpO2
        ax.text(vitals_x, vitals_y - 22, "SpO2", fontsize=10, color=self.colors['blue'])
        ax.text(vitals_x + 12, vitals_y - 22, f"{vitals['spo2']}", fontsize=16, color=self.colors['blue'], fontweight='bold')
        ax.text(vitals_x + 18, vitals_y - 22, "%", fontsize=10, color=self.colors['blue'])
        
        # Temperature
        ax.text(vitals_x, vitals_y - 30, "TEMP", fontsize=10, color=self.colors['text'])
        ax.text(vitals_x + 8, vitals_y - 30, "°C", fontsize=8, color=self.colors['text'])
        ax.text(vitals_x + 5, vitals_y - 34, f"{vitals['temperature']}", fontsize=14, color=self.colors['text'], fontweight='bold')
        
        # Pulse Rate
        ax.text(vitals_x, vitals_y - 40, "PR", fontsize=10, color=self.colors['text'])
        ax.text(vitals_x + 12, vitals_y - 40, str(vitals['pulse_rate']), fontsize=14, color=self.colors['text'], fontweight='bold')
        ax.text(vitals_x + 5, vitals_y - 44, "/MIN", fontsize=8, color=self.colors['text'])
        
        # Menu/Control section
        menu_y = 15
        menu_items = ["SETUP", "TREND", "ALARMS", "FREEZE", "PRINT"]
        for i, item in enumerate(menu_items):
            rect = patches.Rectangle((5 + i*15, menu_y-3), 12, 6, 
                                   linewidth=1, edgecolor=self.colors['text'], 
                                   facecolor='none')
            ax.add_patch(rect)
            ax.text(5 + i*15 + 6, menu_y, item, fontsize=8, color=self.colors['text'], 
                   ha='center', va='center')
        
        # Save the image
        plt.tight_layout()
        plt.savefig(output_path, facecolor=self.colors['bg'], dpi=150, bbox_inches='tight')
        plt.close()
        
        return vitals
    
    def generate_dataset(self, num_images=50, output_dir="generated_monitors"):
        """Generate a dataset of heart monitor images"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        dataset_info = []
        
        for i in range(num_images):
            # Create varied patient scenarios
            patient_scenarios = [
                {"type": "normal", "id": f"RN:{random.randint(10000000, 99999999)}"},
                {"type": "tachycardia", "id": f"RN:{random.randint(10000000, 99999999)}"},
                {"type": "hypertension", "id": f"RN:{random.randint(10000000, 99999999)}"},
                {"type": "low_spo2", "id": f"RN:{random.randint(10000000, 99999999)}"},
            ]
            
            scenario = random.choice(patient_scenarios)
            filename = f"monitor_{i+1:03d}_{scenario['type']}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Generate monitor with specific scenario
            vitals = self.create_monitor_display(filepath, scenario)
            
            # Store metadata
            dataset_info.append({
                "filename": filename,
                "scenario": scenario['type'],
                "patient_id": scenario['id'],
                "vitals": vitals,
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"Generated {filename} - {scenario['type']}")
        
        # Save dataset metadata
        with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\nDataset generation complete!")
        print(f"Generated {num_images} images in '{output_dir}' directory")
        print(f"Metadata saved to '{output_dir}/dataset_info.json'")
        
        return dataset_info

def main():
    """Main function to generate heart monitor images"""
    generator = HeartMonitorGenerator(width=1000, height=700)
    
    # Generate a single example
    print("Generating example heart monitor...")
    vitals = generator.create_monitor_display("example_heart_monitor.png")
    print(f"Example generated with vitals: {vitals}")
    
    # Generate a dataset
    print("\nGenerating dataset...")
    dataset = generator.generate_dataset(num_images=20, output_dir="generated_heart_monitors")
    
    return dataset

if __name__ == "__main__":
    main()
