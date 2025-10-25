import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, HeatMap
import io
import logging
from typing import Dict, Any, Optional, Tuple
from PIL import Image as PILImage, ImageDraw

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Enhanced Configuration
# ------------------------------
class Config:
    PROJECT_TYPES = ["Forestry", "Renewable Energy", "Cookstoves", "Agriculture", "Waste Management", "Methane Capture"]
    COMPLIANCE_STANDARDS = ["Verra VCS", "Gold Standard", "Climate Action Reserve", "American Carbon Registry"]
    VERIFICATION_STATUSES = ["Verified", "Under Review", "Pending Verification", "Requires Correction"]
    
    REGIONS_COORDS = {
        "East Africa": [(-0.0236, 37.9062, "Kenya - Aberdare Forest"), (-6.7924, 39.2083, "Tanzania - Kilimanjaro"), (1.3733, 32.2903, "Uganda - Mabira Forest")],
        "West Africa": [(7.3775, -2.5367, "Ghana - Ashanti"), (9.0579, 8.6753, "Nigeria - Jos Plateau"), (12.2383, -1.5616, "Burkina Faso - Central")],
        "Southeast Asia": [(-0.7893, 113.9213, "Indonesia - Borneo"), (4.5353, 114.7277, "Brunei - Temburong"), (1.2966, 103.7764, "Singapore - Nature Reserves")],
        "South America": [(-3.4653, -62.2159, "Brazil - Amazon"), (-16.2902, -63.5887, "Bolivia - Santa Cruz"), (6.4238, -66.5897, "Venezuela - Orinoco Delta")],
    }

    TAGLINE = "Smart AI. Trusted Climate Impact."
    PRIMARY_COLOR = "#10b981"  # Modern green
    SECONDARY_COLOR = "#3b82f6"  # Blue accent
    BACKGROUND_COLOR = "#f8fafc"
    CARD_BACKGROUND = "#ffffff"
    TEXT_COLOR = "#1e293b"
    
    TYPE_COLORS = {
        "Forestry": "#10b981",
        "Renewable Energy": "#f59e0b",
        "Cookstoves": "#06b6d4",
        "Agriculture": "#84cc16",
        "Waste Management": "#8b5cf6",
        "Methane Capture": "#3b82f6"
    }
    
    RISK_COLORS = {
        "Low": "#10b981",
        "Medium": "#f59e0b",
        "High": "#ef4444"
    }


# ------------------------------
# Custom CSS for Modern UI
# ------------------------------
def inject_custom_css():
    st.markdown("""
        <style>
        /* Main container styling */
        .main {
            background-color: #f8fafc;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            padding: 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .main-title {
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .main-subtitle {
            color: rgba(255, 255, 255, 0.95);
            font-size: 1.2rem;
            margin-top: 0.5rem;
        }
        
        /* Metric cards */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #10b981;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
        }
        
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            padding: 0.5rem 0;
        }
        
        /* Filter section */
        .filter-section {
            background: #f1f5f9;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            transition: all 0.2s;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: white;
            padding: 0.5rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            color: #64748b;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #10b981;
            color: white;
        }
        
        /* Download buttons */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
        }
        
        /* Info boxes */
        .stAlert {
            border-radius: 12px;
            border: none;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        /* Dataframe styling */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Status badges */
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
        }
        
        .status-verified {
            background-color: #d1fae5;
            color: #065f46;
        }
        
        .status-pending {
            background-color: #fef3c7;
            color: #92400e;
        }
        
        /* Chart container */
        .chart-container {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
        }
        
        /* Loading animation */
        .stSpinner > div {
            border-top-color: #10b981 !important;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem 0;
            color: #64748b;
            font-size: 0.875rem;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f5f9;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }
        </style>
    """, unsafe_allow_html=True)


# ------------------------------
# Enhanced Data Management
# ------------------------------
class CarbonProjectDataManager:
    @staticmethod
    @st.cache_data(ttl=3600)
    def generate_realistic_carbon_projects(n_projects: int = 100, seed: Optional[int] = None) -> pd.DataFrame:
        """Generate realistic carbon project data with caching"""
        if seed is None:
            seed = int(datetime.now().timestamp()) % 10000
        np.random.seed(seed)
        
        data = []
        for i in range(n_projects):
            project_id = f"VCS-{2020 + (i % 5)}-{1000+i}"
            project_type = np.random.choice(Config.PROJECT_TYPES)
            region = np.random.choice(list(Config.REGIONS_COORDS.keys()))
            lat, lon, location_name = Config.REGIONS_COORDS[region][np.random.randint(0, len(Config.REGIONS_COORDS[region]))]
            
            lat += np.random.uniform(-0.5, 0.5)
            lon += np.random.uniform(-0.5, 0.5)
            
            baseline_emissions = np.random.randint(5000, 50000)
            reduction_rate = np.random.uniform(0.15, 0.85)
            credits_generated = int(baseline_emissions * reduction_rate)
            
            buffer_percentage = np.random.uniform(0.10, 0.20)
            buffer_credits = int(credits_generated * buffer_percentage)
            net_credits = credits_generated - buffer_credits
            
            permanence_risk = np.random.choice(["Low", "Medium", "High"], p=[0.5, 0.35, 0.15])
            additionality_score = np.random.uniform(65, 98)
            verification_status = np.random.choice(Config.VERIFICATION_STATUSES, p=[0.6, 0.2, 0.15, 0.05])
            compliance_standard = np.random.choice(Config.COMPLIANCE_STANDARDS)
            
            start_date = datetime.now() - timedelta(days=np.random.randint(365, 365*5))
            vintage_year = start_date.year + np.random.randint(0, 5)
            
            # Additional metrics
            co2_price = np.random.uniform(15, 85)
            market_value = net_credits * co2_price
            
            data.append([
                project_id, project_type, region, location_name,
                credits_generated, net_credits, buffer_credits, baseline_emissions,
                np.random.randint(1, 10), verification_status, compliance_standard,
                permanence_risk, additionality_score, vintage_year,
                start_date, datetime.now(), lat, lon, co2_price, market_value
            ])
        
        columns = [
            "ProjectID", "Type", "Region", "Location", 
            "TotalCredits", "NetCredits", "BufferCredits", "BaselineEmissions",
            "MonitoringPeriod", "VerificationStatus", "ComplianceStandard",
            "PermanenceRisk", "AdditionalityScore", "VintageYear",
            "StartDate", "LastUpdate", "Latitude", "Longitude", "CO2Price", "MarketValue"
        ]
        return pd.DataFrame(data, columns=columns)

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced data validation"""
        issues = []
        warnings = []
        
        if df.empty:
            issues.append("DataFrame is empty")
            return {"valid": False, "issues": issues, "warnings": warnings}
        
        # Critical validations
        if not df['NetCredits'].ge(0).all():
            issues.append("Negative NetCredits detected")
        if not df['AdditionalityScore'].between(0, 100).all():
            issues.append("AdditionalityScore outside valid range [0,100]")
        if not (df['BufferCredits'] <= df['TotalCredits']).all():
            issues.append("BufferCredits exceed TotalCredits")
        if df[['Latitude', 'Longitude']].isnull().any().any():
            issues.append("Missing geographic coordinates")
        
        # Warning validations
        if df['NetCredits'].median() == 0:
            warnings.append("Median NetCredits is zero")
        if (df['AdditionalityScore'] < 70).sum() > len(df) * 0.3:
            warnings.append("Over 30% of projects have low additionality scores (<70)")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "total_projects": len(df),
            "verified_count": len(df[df['VerificationStatus'] == 'Verified'])
        }


# ------------------------------
# Enhanced Filtering
# ------------------------------
@st.cache_data
def apply_filters(
    df: pd.DataFrame,
    regions: list,
    types: list,
    statuses: list,
    risks: list,
    credit_range: Tuple[int, int],
    additionality_range: Tuple[float, float],
    vintage_range: Tuple[int, int]
) -> pd.DataFrame:
    """Apply multiple filters with caching"""
    filtered = df.copy()
    
    if regions:
        filtered = filtered[filtered["Region"].isin(regions)]
    if types:
        filtered = filtered[filtered["Type"].isin(types)]
    if statuses:
        filtered = filtered[filtered["VerificationStatus"].isin(statuses)]
    if risks:
        filtered = filtered[filtered["PermanenceRisk"].isin(risks)]
    
    filtered = filtered[
        filtered["NetCredits"].between(credit_range[0], credit_range[1]) &
        filtered["AdditionalityScore"].between(additionality_range[0], additionality_range[1]) &
        filtered["VintageYear"].between(vintage_range[0], vintage_range[1])
    ]
    
    return filtered


# ------------------------------
# Enhanced Visualization Manager
# ------------------------------
class VisualizationManager:
    @staticmethod
    def _apply_modern_theme(fig):
        """Apply modern, clean theme to figures"""
        fig.update_layout(
            font=dict(family="Inter, system-ui, sans-serif", size=12, color="#1e293b"),
            title=dict(font=dict(size=18, color="#0f172a", family="Inter"), x=0.5, xanchor='center'),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(
                showgrid=True,
                gridcolor="#f1f5f9",
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor="#e2e8f0"
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#f1f5f9",
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor="#e2e8f0"
            ),
            margin=dict(l=60, r=40, t=80, b=60),
            hovermode="closest",
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Inter"
            )
        )
        return fig

    @staticmethod
    def credits_by_type_enhanced(df):
        """Enhanced bar chart with better styling"""
        if df.empty:
            return go.Figure().add_annotation(
                text="No data available",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="#64748b")
            )
        
        # Aggregate data
        agg_df = df.groupby(['Region', 'Type'])['NetCredits'].sum().reset_index()
        
        fig = px.bar(
            agg_df,
            x="Type",
            y="NetCredits",
            color="Type",
            facet_col="Region",
            facet_col_wrap=2,
            title="Net Carbon Credits by Project Type & Region",
            labels={"NetCredits": "Net Credits (tCO₂e)", "Type": "Project Type"},
            color_discrete_map=Config.TYPE_COLORS,
            height=600
        )
        
        fig.update_traces(
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>Net Credits: %{y:,.0f} tCO₂e<extra></extra>"
        )
        
        fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
        
        # Update facet titles
        for i, region in enumerate(agg_df["Region"].unique()):
            fig.layout.annotations[i].update(
                text=f"<b>{region}</b>",
                font=dict(size=14, color="#0f172a")
            )
        
        return VisualizationManager._apply_modern_theme(fig)

    @staticmethod
    def compliance_donut(df):
        """Modern donut chart for compliance standards"""
        if df.empty:
            return go.Figure().add_annotation(
                text="No data available",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="#64748b")
            )
        
        compliance_counts = df['ComplianceStandard'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=compliance_counts.index,
            values=compliance_counts.values,
            hole=0.5,
            marker=dict(
                colors=['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b'],
                line=dict(color='white', width=3)
            ),
            textposition='outside',
            textinfo='label+percent',
            hovertemplate="<b>%{label}</b><br>Projects: %{value}<br>Percentage: %{percent}<extra></extra>"
        )])
        
        fig.update_layout(
            title="Compliance Standards Distribution",
            showlegend=False,
            annotations=[dict(
                text=f'{len(df)}<br>Projects',
                x=0.5, y=0.5,
                font_size=20,
                showarrow=False,
                font=dict(color="#0f172a")
            )]
        )
        
        return VisualizationManager._apply_modern_theme(fig)

    @staticmethod
    def risk_bubble_chart(df, max_points=500):
        """Enhanced bubble chart with risk analysis"""
        if df.empty:
            return go.Figure().add_annotation(
                text="No data available",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="#64748b")
            )
        
        df_plot = df.copy()
        if len(df_plot) > max_points:
            df_plot = df_plot.sample(n=max_points, random_state=42)
        
        df_plot['PermanenceRisk'] = pd.Categorical(
            df_plot['PermanenceRisk'],
            categories=["Low", "Medium", "High"],
            ordered=True
        )
        
        fig = px.scatter(
            df_plot,
            x="AdditionalityScore",
            y="NetCredits",
            size="MarketValue",
            color="PermanenceRisk",
            hover_data={
                "ProjectID": True,
                "Type": True,
                "Region": True,
                "MarketValue": ":,.0f",
                "AdditionalityScore": ":.1f",
                "NetCredits": ":,.0f",
                "PermanenceRisk": True
            },
            title="Risk Profile: Additionality vs. Net Credits",
            labels={
                "AdditionalityScore": "Additionality Score (%)",
                "NetCredits": "Net Credits (tCO₂e)",
                "MarketValue": "Market Value (USD)"
            },
            color_discrete_map=Config.RISK_COLORS,
            size_max=60
        )
        
        fig.update_traces(
            marker=dict(
                opacity=0.7,
                line=dict(width=1, color='white')
            )
        )
        
        return VisualizationManager._apply_modern_theme(fig)

    @staticmethod
    def timeline_chart(df):
        """Timeline of credits by vintage year"""
        if df.empty:
            return go.Figure()
        
        timeline_df = df.groupby(['VintageYear', 'Type'])['NetCredits'].sum().reset_index()
        
        fig = px.bar(
            timeline_df,
            x='VintageYear',
            y='NetCredits',
            color='Type',
            title='Carbon Credits Generation Timeline',
            labels={'NetCredits': 'Net Credits (tCO₂e)', 'VintageYear': 'Vintage Year'},
            color_discrete_map=Config.TYPE_COLORS,
            barmode='stack'
        )
        
        return VisualizationManager._apply_modern_theme(fig)

    @staticmethod
    def regional_heatmap(df):
        """Heatmap of regional performance"""
        if df.empty:
            return go.Figure()
        
        pivot_df = df.groupby(['Region', 'Type'])['NetCredits'].sum().reset_index()
        pivot_table = pivot_df.pivot(index='Region', columns='Type', values='NetCredits').fillna(0)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='Greens',
            hoverongaps=False,
            hovertemplate='Region: %{y}<br>Type: %{x}<br>Credits: %{z:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Regional Performance Heatmap',
            xaxis_title='Project Type',
            yaxis_title='Region'
        )
        
        return VisualizationManager._apply_modern_theme(fig)

    @staticmethod
    def create_enhanced_popup(row) -> str:
        """Create beautiful popup for map markers"""
        status_color = "#10b981" if row['VerificationStatus'] == 'Verified' else "#f59e0b"
        risk_color = Config.RISK_COLORS.get(row['PermanenceRisk'], "#64748b")
        
        return f"""
        <div style="font-family: Inter, sans-serif; width: 280px;">
            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                        color: white; padding: 12px; border-radius: 8px 8px 0 0; margin: -10px -10px 10px -10px;">
                <h4 style="margin: 0; font-size: 16px;">{row['ProjectID']}</h4>
                <p style="margin: 4px 0 0 0; font-size: 12px; opacity: 0.9;">{row['Type']}</p>
            </div>
            <div style="padding: 0 4px;">
                <p style="margin: 8px 0; font-size: 13px;">
                    <strong>📍 Location:</strong> {row['Location']}<br>
                    <strong>💰 Net Credits:</strong> {row['NetCredits']:,.0f} tCO₂e<br>
                    <strong>💵 Market Value:</strong> ${row['MarketValue']:,.0f}<br>
                    <strong>📅 Vintage:</strong> {row['VintageYear']}<br>
                    <strong>✅ Status:</strong> <span style="color: {status_color}; font-weight: 600;">{row['VerificationStatus']}</span><br>
                    <strong>⚠️ Risk:</strong> <span style="color: {risk_color}; font-weight: 600;">{row['PermanenceRisk']}</span><br>
                    <strong>📊 Additionality:</strong> {row['AdditionalityScore']:.1f}%
                </p>
            </div>
        </div>
        """


# ------------------------------
# Enhanced Report Generator
# ------------------------------
class ReportGenerator:
    @staticmethod
    def fig_to_png_bytes(fig) -> bytes:
        """Convert Plotly figure to PNG bytes"""
        buf = io.BytesIO()
        try:
            fig.write_image(buf, format="png", width=900, height=550, scale=2)
        except Exception as e:
            logger.warning(f"Image export failed: {e}")
            img = PILImage.new('RGB', (900, 550), color=(248, 250, 252))
            draw = ImageDraw.Draw(img)
            draw.text((50, 270), "Chart visualization unavailable", fill=(100, 100, 100))
            img.save(buf, format="PNG")
        buf.seek(0)
        return buf.read()

    @staticmethod
    def _create_color_legend():
        """Create color-coded project type legend"""
        data = [["Project Type", "Color"]]
        col_widths = [2.8 * inch, 0.7 * inch]
        
        style = [
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.Color(0.06, 0.73, 0.51)),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]
        
        for ptype in Config.PROJECT_TYPES:
            data.append([ptype, ""])
            hex_color = Config.TYPE_COLORS.get(ptype, "#64748b")
            r = int(hex_color[1:3], 16) / 255.0
            g = int(hex_color[3:5], 16) / 255.0
            b = int(hex_color[5:7], 16) / 255.0
            row_idx = len(data) - 1
            style.append(('BACKGROUND', (1, row_idx), (1, row_idx), rl_colors.Color(r, g, b)))
        
        table = Table(data, colWidths=col_widths, rowHeights=22)
        table.setStyle(TableStyle(style))
        return table

    @staticmethod
    def generate_pdf(df: pd.DataFrame, figures: Dict[str, go.Figure]) -> Optional[bytes]:
        """Generate comprehensive PDF report"""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                topMargin=0.6 * inch,
                bottomMargin=0.6 * inch,
                leftMargin=0.7 * inch,
                rightMargin=0.7 * inch
            )
            
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=26,
                textColor=rl_colors.Color(0.06, 0.73, 0.51),
                spaceAfter=8,
                alignment=1,
                fontName='Helvetica-Bold'
            )
            
            centered_style = ParagraphStyle(
                'Centered',
                parent=styles['Normal'],
                alignment=1,
                fontSize=12,
                leading=16
            )
            
            story = []

            # Header
            story.append(Paragraph("GreenScope Analytics", title_style))
            story.append(Paragraph("<b>Smart AI. Trusted Climate Impact.</b>", centered_style))
            story.append(Spacer(1, 24))

            # Executive Summary
            story.append(Paragraph("<b>Executive Summary</b>", styles['Heading1']))
            story.append(Spacer(1, 8))
            
            total_market_value = df['MarketValue'].sum()
            avg_price = df['CO2Price'].mean()
            
            summary_data = [
                ["Total Active Projects", f"{len(df):,}"],
                ["Total Net Credits Generated", f"{df['NetCredits'].sum():,.0f} tCO₂e"],
                ["Total Market Value", f"${total_market_value:,.0f}"],
                ["Average CO₂ Price", f"${avg_price:.2f}/tCO₂e"],
                ["Average Additionality Score", f"{df['AdditionalityScore'].mean():.1f}%"],
                ["Verified Projects", f"{len(df[df['VerificationStatus']=='Verified'])} ({(len(df[df['VerificationStatus']=='Verified'])/len(df)*100):.1f}%)"],
                ["Low Risk Projects", f"{len(df[df['PermanenceRisk']=='Low'])} ({(len(df[df['PermanenceRisk']=='Low'])/len(df)*100):.1f}%)"],
                ["Portfolio Vintage Range", f"{df['VintageYear'].min()} – {df['VintageYear'].max()}"]
            ]
            
            summary_table = Table(summary_data, colWidths=[3.2*inch, 2.2*inch])
            summary_table.setStyle(TableStyle([
                ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
                ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
                ('FONTSIZE', (0,0), (-1,-1), 11),
                ('TOPPADDING', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                ('LINEBELOW', (0,0), (-1,-2), 0.5, rl_colors.Color(0.9, 0.9, 0.9)),
                ('LINEBELOW', (0,-1), (-1,-1), 1.5, rl_colors.Color(0.06, 0.73, 0.51)),
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 20))

            # Project Type Legend
            story.append(Paragraph("<b>Project Type Legend</b>", styles['Heading2']))
            story.append(Spacer(1, 8))
            story.append(ReportGenerator._create_color_legend())
            story.append(Spacer(1, 24))

            # Visualizations
            chart_info = [
                ("Net Carbon Credits by Project Type & Region", figures.get("credits")),
                ("Compliance Standards Distribution", figures.get("compliance")),
                ("Risk Profile Analysis", figures.get("risk")),
                ("Carbon Credits Generation Timeline", figures.get("timeline"))
            ]
            
            for title, fig in chart_info:
                if fig is not None:
                    story.append(Paragraph(f"<b>{title}</b>", styles['Heading2']))
                    story.append(Spacer(1, 8))
                    img_bytes = ReportGenerator.fig_to_png_bytes(fig)
                    img = Image(io.BytesIO(img_bytes), width=6.8 * inch, height=4.2 * inch)
                    story.append(img)
                    story.append(Spacer(1, 20))

            # Top Projects Table
            story.append(Paragraph("<b>Top 10 Projects by Net Credits</b>", styles['Heading1']))
            story.append(Spacer(1, 10))
            
            if not df.empty:
                top = df.nlargest(10, 'NetCredits')[['ProjectID', 'Type', 'NetCredits', 'MarketValue', 'VerificationStatus']]
                table_data = [['Project ID', 'Type', 'Net Credits', 'Market Value', 'Status']]
                
                for _, row in top.iterrows():
                    table_data.append([
                        str(row['ProjectID']),
                        str(row['Type']),
                        f"{row['NetCredits']:,.0f}",
                        f"${row['MarketValue']:,.0f}",
                        str(row['VerificationStatus'])
                    ])
                
                top_table = Table(table_data, colWidths=[1.3*inch, 1.3*inch, 1.2*inch, 1.2*inch, 1.4*inch])
                top_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), rl_colors.Color(0.06, 0.73, 0.51)),
                    ('TEXTCOLOR', (0,0), (-1,0), rl_colors.white),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,0), 11),
                    ('BOTTOMPADDING', (0,0), (-1,0), 10),
                    ('TOPPADDING', (0,0), (-1,0), 10),
                    ('BACKGROUND', (0,1), (-1,-1), rl_colors.Color(0.97, 0.98, 0.97)),
                    ('GRID', (0,0), (-1,-1), 0.8, rl_colors.Color(0.85, 0.85, 0.85)),
                    ('FONTSIZE', (0,1), (-1,-1), 10),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [rl_colors.white, rl_colors.Color(0.97, 0.98, 0.97)])
                ]))
                story.append(top_table)

            story.append(Spacer(1, 30))
            story.append(Paragraph(
                f"<i>© {datetime.now().year} GreenScope Analytics — Climate Intelligence Platform</i>",
                centered_style
            ))
            story.append(Paragraph(
                f"<i>Report Generated: {datetime.now().strftime('%B %d, %Y at %H:%M UTC')}</i>",
                centered_style
            ))

            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            logger.exception("PDF generation failed")
            return None


# ------------------------------
# Enhanced Main Application
# ------------------------------
def render_header():
    """Render modern header with branding"""
    st.markdown("""
        <div class="main-header">
            <div class="main-title">🌍 GreenScope Analytics</div>
            <div class="main-subtitle">Smart AI. Trusted Climate Impact.</div>
        </div>
    """, unsafe_allow_html=True)


def render_kpi_cards(df: pd.DataFrame):
    """Render beautiful KPI metric cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Projects",
            f"{len(df):,}",
            delta=f"+{len(df[df['VerificationStatus']=='Verified'])} verified",
            delta_color="normal"
        )
    
    with col2:
        total_credits = df['NetCredits'].sum()
        st.metric(
            "Net Credits",
            f"{total_credits:,.0f}",
            delta="tCO₂e",
            delta_color="off"
        )
    
    with col3:
        avg_additionality = df['AdditionalityScore'].mean()
        st.metric(
            "Avg Additionality",
            f"{avg_additionality:.1f}%",
            delta=f"±{df['AdditionalityScore'].std():.1f}%",
            delta_color="off"
        )
    
    with col4:
        total_value = df['MarketValue'].sum()
        st.metric(
            "Market Value",
            f"${total_value/1e6:.1f}M",
            delta=f"${df['CO2Price'].mean():.0f}/tCO₂e avg",
            delta_color="off"
        )


def render_advanced_filters():
    """Render enhanced filter sidebar"""
    st.sidebar.markdown("## 🔍 Advanced Filters")
    
    with st.sidebar.expander("📍 Geographic Filters", expanded=True):
        regions = st.multiselect(
            "Regions",
            sorted(st.session_state.df["Region"].unique()),
            default=sorted(st.session_state.df["Region"].unique()),
            key="region_filter"
        )
    
    with st.sidebar.expander("🏷️ Project Filters", expanded=True):
        types = st.multiselect(
            "Project Types",
            sorted(st.session_state.df["Type"].unique()),
            default=sorted(st.session_state.df["Type"].unique()),
            key="type_filter"
        )
        
        statuses = st.multiselect(
            "Verification Status",
            Config.VERIFICATION_STATUSES,
            default=Config.VERIFICATION_STATUSES,
            key="status_filter"
        )
        
        risks = st.multiselect(
            "Permanence Risk",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
            key="risk_filter"
        )
    
    with st.sidebar.expander("📊 Metric Filters", expanded=True):
        credit_range = st.slider(
            "Net Credits (tCO₂e)",
            int(st.session_state.df["NetCredits"].min()),
            int(st.session_state.df["NetCredits"].max()),
            (int(st.session_state.df["NetCredits"].min()), 
             int(st.session_state.df["NetCredits"].max())),
            key="credit_range"
        )
        
        additionality_range = st.slider(
            "Additionality Score (%)",
            float(st.session_state.df["AdditionalityScore"].min()),
            100.0,
            (float(st.session_state.df["AdditionalityScore"].min()), 100.0),
            key="additionality_range"
        )
        
        vintage_range = st.slider(
            "Vintage Year",
            int(st.session_state.df["VintageYear"].min()),
            int(st.session_state.df["VintageYear"].max()),
            (int(st.session_state.df["VintageYear"].min()),
             int(st.session_state.df["VintageYear"].max())),
            key="vintage_range"
        )
    
    return regions, types, statuses, risks, credit_range, additionality_range, vintage_range


def render_dashboard_tab(df: pd.DataFrame):
    """Render enhanced dashboard with insights"""
    st.markdown("### 📊 Portfolio Overview")
    
    render_kpi_cards(df)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Regional Distribution")
        regional_df = df.groupby('Region').agg({
            'NetCredits': 'sum',
            'ProjectID': 'count',
            'MarketValue': 'sum'
        }).reset_index()
        regional_df.columns = ['Region', 'Total Credits', 'Project Count', 'Market Value']
        
        fig = px.bar(
            regional_df,
            x='Region',
            y='Total Credits',
            color='Market Value',
            title='',
            color_continuous_scale='Greens'
        )
        fig = VisualizationManager._apply_modern_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Risk Distribution")
        risk_counts = df['PermanenceRisk'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.6,
            marker=dict(colors=[Config.RISK_COLORS[risk] for risk in risk_counts.index]),
            textinfo='label+percent'
        )])
        fig = VisualizationManager._apply_modern_theme(fig)
        fig.update_layout(showlegend=False, height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Quick insights
    st.markdown("#### 💡 Key Insights")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        top_type = df.groupby('Type')['NetCredits'].sum().idxmax()
        top_credits = df.groupby('Type')['NetCredits'].sum().max()
        st.info(f"**Leading Project Type**\n\n{top_type} with {top_credits:,.0f} tCO₂e")
    
    with insight_col2:
        verified_pct = (len(df[df['VerificationStatus']=='Verified']) / len(df)) * 100
        st.success(f"**Verification Rate**\n\n{verified_pct:.1f}% of projects verified")
    
    with insight_col3:
        low_risk_pct = (len(df[df['PermanenceRisk']=='Low']) / len(df)) * 100
        st.success(f"**Low Risk Portfolio**\n\n{low_risk_pct:.1f}% rated low risk")


def render_analytics_tab(df: pd.DataFrame):
    """Render comprehensive analytics"""
    st.markdown("### 📈 Advanced Analytics")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        fig1 = VisualizationManager.credits_by_type_enhanced(df)
        st.plotly_chart(fig1, use_container_width=True)
    
    with viz_col2:
        fig2 = VisualizationManager.compliance_donut(df)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    fig3 = VisualizationManager.risk_bubble_chart(df)
    st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig4 = VisualizationManager.timeline_chart(df)
        st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        fig5 = VisualizationManager.regional_heatmap(df)
        st.plotly_chart(fig5, use_container_width=True)


def render_map_tab(df: pd.DataFrame):
    """Render interactive map"""
    st.markdown("### 🗺️ Geographic Distribution")
    
    if df.empty:
        st.info("No projects to display on map")
        return
    
    # Map controls
    col1, col2, col3 = st.columns(3)
    with col1:
        show_heatmap = st.checkbox("Show Heatmap", value=False)
    with col2:
        cluster_markers = st.checkbox("Cluster Markers", value=True)
    with col3:
        color_by = st.selectbox("Color By", ["Verification Status", "Risk Level", "Project Type"])
    
    # Create map
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=3,
        tiles="CartoDB positron"
    )
    
    # Add heatmap layer if selected
    if show_heatmap:
        heat_data = [[row['Latitude'], row['Longitude'], row['NetCredits']/1000] 
                     for _, row in df.iterrows()]
        HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)
    
    # Add markers
    if cluster_markers:
        marker_cluster = MarkerCluster().add_to(m)
        marker_parent = marker_cluster
    else:
        marker_parent = m
    
    for _, row in df.iterrows():
        # Determine marker color based on selection
        if color_by == "Verification Status":
            color = "green" if row["VerificationStatus"] == "Verified" else "orange"
        elif color_by == "Risk Level":
            risk_color_map = {"Low": "green", "Medium": "orange", "High": "red"}
            color = risk_color_map.get(row["PermanenceRisk"], "blue")
        else:  # Project Type
            color = "green"
        
        popup_html = VisualizationManager.create_enhanced_popup(row)
        
        folium.Marker(
            [row["Latitude"], row["Longitude"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['ProjectID']} ({row['Type']})",
            icon=folium.Icon(color=color, icon="leaf", prefix="fa")
        ).add_to(marker_parent)
    
    st_folium(m, width="100%", height=600)
    
    # Map statistics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Locations", len(df))
    with col2:
        st.metric("Regions Covered", df['Region'].nunique())
    with col3:
        st.metric("Countries", df['Location'].apply(lambda x: x.split(' - ')[0]).nunique())


def render_compare_tab(df: pd.DataFrame):
    """Render project comparison"""
    st.markdown("### ⚖️ Project Comparison")
    
    project_ids = sorted(df["ProjectID"].unique())
    
    if len(project_ids) < 2:
        st.warning("Need at least 2 projects to compare")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        p1 = st.selectbox("Select First Project", project_ids, key="compare_p1")
    with col2:
        available_projects = [p for p in project_ids if p != p1]
        p2 = st.selectbox("Select Second Project", available_projects, key="compare_p2")
    
    if p1 and p2 and p1 != p2:
        d1 = df[df["ProjectID"] == p1].iloc[0]
        d2 = df[df["ProjectID"] == p2].iloc[0]
        
        # Project Cards
        card_col1, card_col2 = st.columns(2)
        
        with card_col1:
            st.markdown(f"""
                <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #10b981;">
                    <h3 style="color: #0f172a; margin-top: 0;">{d1['ProjectID']}</h3>
                    <p><strong>Type:</strong> {d1['Type']}</p>
                    <p><strong>Location:</strong> {d1['Location']}</p>
                    <p><strong>Region:</strong> {d1['Region']}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Metrics")
            st.metric("Net Credits", f"{d1['NetCredits']:,.0f} tCO₂e")
            st.metric("Market Value", f"${d1['MarketValue']:,.0f}")
            st.metric("Additionality Score", f"{d1['AdditionalityScore']:.1f}%")
            st.metric("CO₂ Price", f"${d1['CO2Price']:.2f}")
            
            st.markdown("#### Status")
            st.write(f"**Verification:** {d1['VerificationStatus']}")
            st.write(f"**Risk Level:** {d1['PermanenceRisk']}")
            st.write(f"**Standard:** {d1['ComplianceStandard']}")
            st.write(f"**Vintage:** {d1['VintageYear']}")
        
        with card_col2:
            st.markdown(f"""
                <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #3b82f6;">
                    <h3 style="color: #0f172a; margin-top: 0;">{d2['ProjectID']}</h3>
                    <p><strong>Type:</strong> {d2['Type']}</p>
                    <p><strong>Location:</strong> {d2['Location']}</p>
                    <p><strong>Region:</strong> {d2['Region']}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Metrics")
            delta_credits = d2['NetCredits'] - d1['NetCredits']
            st.metric("Net Credits", f"{d2['NetCredits']:,.0f} tCO₂e", 
                     delta=f"{delta_credits:+,.0f}")
            
            delta_value = d2['MarketValue'] - d1['MarketValue']
            st.metric("Market Value", f"${d2['MarketValue']:,.0f}",
                     delta=f"${delta_value:+,.0f}")
            
            delta_add = d2['AdditionalityScore'] - d1['AdditionalityScore']
            st.metric("Additionality Score", f"{d2['AdditionalityScore']:.1f}%",
                     delta=f"{delta_add:+.1f}%")
            
            delta_price = d2['CO2Price'] - d1['CO2Price']
            st.metric("CO₂ Price", f"${d2['CO2Price']:.2f}",
                     delta=f"${delta_price:+.2f}")
            
            st.markdown("#### Status")
            st.write(f"**Verification:** {d2['VerificationStatus']}")
            st.write(f"**Risk Level:** {d2['PermanenceRisk']}")
            st.write(f"**Standard:** {d2['ComplianceStandard']}")
            st.write(f"**Vintage:** {d2['VintageYear']}")
        
        st.markdown("---")
        st.markdown("### 📊 Comparative Analysis")
        
        # Comparison chart
        comp_df = pd.DataFrame({
            'Project': [p1, p2],
            'Net Credits': [d1['NetCredits'], d2['NetCredits']],
            'Market Value ($)': [d1['MarketValue'], d2['MarketValue']],
            'Additionality (%)': [d1['AdditionalityScore'], d2['AdditionalityScore']],
            'Buffer Credits': [d1['BufferCredits'], d2['BufferCredits']]
        })
        
        fig = go.Figure()
        
        metrics = ['Net Credits', 'Market Value ($)', 'Additionality (%)', 'Buffer Credits']
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=comp_df['Project'],
                y=comp_df[metric],
                text=comp_df[metric],
                texttemplate='%{text:,.0f}',
                textposition='outside'
            ))
        
        fig.update_layout(
            title='Side-by-Side Comparison',
            barmode='group',
            height=500
        )
        fig = VisualizationManager._apply_modern_theme(fig)
        
        st.plotly_chart(fig, use_container_width=True)


def render_report_tab(df: pd.DataFrame):
    """Render report generation interface"""
    st.markdown("### 📋 Generate Portfolio Report")
    
    st.info("📄 Generate a comprehensive PDF report with all visualizations, analytics, and project summaries.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Report Configuration")
        
        include_map = st.checkbox("Include Geographic Map", value=True)
        include_top_projects = st.checkbox("Include Top 10 Projects Table", value=True)
        include_timeline = st.checkbox("Include Timeline Analysis", value=True)
        
        st.markdown("#### Report Statistics")
        st.write(f"- **Projects in Report:** {len(df)}")
        st.write(f"- **Total Credits:** {df['NetCredits'].sum():,.0f} tCO₂e")
        st.write(f"- **Total Value:** ${df['MarketValue'].sum():,.0f}")
        st.write(f"- **Date Range:** {df['VintageYear'].min()} - {df['VintageYear'].max()}")
    
    with col2:
        st.markdown("#### Quick Export")
        
        # CSV Export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download CSV Data",
            data=csv,
            file_name=f"greenscope_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Excel Export (simplified)
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Projects', index=False)
            
            summary_df = pd.DataFrame({
                'Metric': ['Total Projects', 'Total Credits', 'Total Value', 'Avg Additionality'],
                'Value': [len(df), df['NetCredits'].sum(), df['MarketValue'].sum(), df['AdditionalityScore'].mean()]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        excel_buffer.seek(0)
        st.download_button(
            "📊 Download Excel Report",
            data=excel_buffer,
            file_name=f"greenscope_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # PDF Generation
    st.markdown("#### 📄 Generate Branded PDF Report")
    
    if st.button("🎨 Generate Professional PDF", type="primary", use_container_width=True):
        with st.spinner("Generating comprehensive report... This may take a moment."):
            # Generate all figures
            fig1 = VisualizationManager.credits_by_type_enhanced(df)
            fig2 = VisualizationManager.compliance_donut(df)
            fig3 = VisualizationManager.risk_bubble_chart(df)
            fig4 = VisualizationManager.timeline_chart(df) if include_timeline else None
            
            figures = {
                "credits": fig1,
                "compliance": fig2,
                "risk": fig3,
                "timeline": fig4
            }
            
            pdf_bytes = ReportGenerator.generate_pdf(df, figures)
            
            if pdf_bytes:
                st.success("✅ Report generated successfully!")
                st.download_button(
                    "📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"GreenScope_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.error("❌ Failed to generate report. Please check logs for details.")


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="GreenScope Analytics | Climate Intelligence Platform",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS
    inject_custom_css()
    
    # Render header
    render_header()
    
    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ Configuration")
    
    with st.sidebar.expander("🔧 Data Settings", expanded=False):
        fixed_dataset = st.checkbox("Use Fixed Dataset (seed=42)", value=True)
        n_projects = st.slider("Number of Projects", 50, 200, 100)
        
        if st.button("🔄 Regenerate Data"):
            st.session_state.pop('df', None)
            st.rerun()
    
    # Initialize or load data
    if 'df' not in st.session_state:
        with st.spinner("🔄 Loading portfolio data..."):
            seed = 42 if fixed_dataset else None
            st.session_state.df = CarbonProjectDataManager.generate_realistic_carbon_projects(
                n_projects=n_projects,
                seed=seed
            )
    
    df = st.session_state.df
    
    # Validate data
    validation = CarbonProjectDataManager.validate_dataframe(df)
    
    if not validation["valid"]:
        st.error("⚠️ Data Validation Issues Detected:")
        for issue in validation["issues"]:
            st.error(f"- {issue}")
        return
    
    if validation.get("warnings"):
        with st.expander("⚠️ Data Warnings", expanded=False):
            for warning in validation["warnings"]:
                st.warning(warning)
    
    # Apply filters
    regions, types, statuses, risks, credit_range, additionality_range, vintage_range = render_advanced_filters()
    
    # Reset filters button
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        for key in list(st.session_state.keys()):
            if 'filter' in key or 'range' in key:
                del st.session_state[key]
        st.rerun()
    
    # Filter data
    filtered_df = apply_filters(
        df, regions, types, statuses, risks,
        credit_range, additionality_range, vintage_range
    )
    
    # Show filter results
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 📊 Filtered Results")
    st.sidebar.metric("Projects Shown", f"{len(filtered_df):,}")
    st.sidebar.metric("Total Credits", f"{filtered_df['NetCredits'].sum():,.0f}")
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No projects match your current filters. Please adjust your criteria.")
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard",
        "📈 Analytics",
        "🗺️ Geographic Map",
        "⚖️ Compare Projects",
        "📋 Reports",
        "📑 Data Table"
    ])
    
    with tab1:
        render_dashboard_tab(filtered_df)
    
    with tab2:
        render_analytics_tab(filtered_df)
    
    with tab3:
        render_map_tab(filtered_df)
    
    with tab4:
        render_compare_tab(filtered_df)
    
    with tab5:
        render_report_tab(filtered_df)
    
    with tab6:
        st.markdown("### 📑 Project Data Table")
        
        # Column selector
        all_columns = filtered_df.columns.tolist()
        default_columns = ['ProjectID', 'Type', 'Region', 'NetCredits', 'VerificationStatus', 
                          'PermanenceRisk', 'AdditionalityScore', 'MarketValue']
        
        selected_columns = st.multiselect(
            "Select Columns to Display",
            all_columns,
            default=[col for col in default_columns if col in all_columns]
        )
        
        if selected_columns:
            # Search functionality
            search_term = st.text_input("🔍 Search Projects", "")
            
            display_df = filtered_df[selected_columns].copy()
            
            if search_term:
                mask = display_df.astype(str).apply(
                    lambda row: row.str.contains(search_term, case=False, na=False).any(),
                    axis=1
                )
                display_df = display_df[mask]
            
            # Sorting
            col1, col2 = st.columns([3, 1])
            with col1:
                sort_by = st.selectbox("Sort By", selected_columns)
            with col2:
                sort_order = st.selectbox("Order", ["Descending", "Ascending"])
            
            display_df = display_df.sort_values(
                by=sort_by,
                ascending=(sort_order == "Ascending")
            )
            
            # Display table with formatting
            # Build format dictionary only for columns that exist
            format_dict = {}
            format_specs = {
                'NetCredits': '{:,.0f}',
                'TotalCredits': '{:,.0f}',
                'BufferCredits': '{:,.0f}',
                'BaselineEmissions': '{:,.0f}',
                'AdditionalityScore': '{:.1f}%',
                'MarketValue': '${:,.0f}',
                'CO2Price': '${:.2f}'
            }
            
            for col_name, fmt in format_specs.items():
                if col_name in display_df.columns:
                    format_dict[col_name] = fmt
            
            st.dataframe(
                display_df.style.format(format_dict),
                use_container_width=True,
                height=500
            )
            
            st.caption(f"Showing {len(display_df)} of {len(filtered_df)} projects")
        else:
            st.info("Please select at least one column to display")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
        <div class="footer">
            <p><strong>GreenScope Analytics</strong> — Climate Intelligence Platform</p>
            <p>Showing {len(filtered_df):,} of {len(df):,} projects | 
               Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} | 
               Total Portfolio Value: ${filtered_df['MarketValue'].sum():,.0f}</p>
            <p style="font-size: 0.75rem; margin-top: 1rem;">
                © {datetime.now().year} GreenScope Analytics. All rights reserved. | 
                <a href="#" style="color: #10b981; text-decoration: none;">Privacy Policy</a> | 
                <a href="#" style="color: #10b981; text-decoration: none;">Terms of Service</a>
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()