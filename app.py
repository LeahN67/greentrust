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
from folium.plugins import MarkerCluster
import io
import base64
import logging
from typing import Dict, Any, Optional
from PIL import Image as PILImage, ImageDraw

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Configuration
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
    PRIMARY_COLOR = "#2E8B57"

    TYPE_COLORS = {
        "Forestry": "#2E8B57",
        "Renewable Energy": "#FF6F61",
        "Cookstoves": "#6ECEDA",
        "Agriculture": "#FFD166",
        "Waste Management": "#06D6A0",
        "Methane Capture": "#118AB2"
    }


# ------------------------------
# Data Management
# ------------------------------
class CarbonProjectDataManager:
    @staticmethod
    def generate_realistic_carbon_projects(n_projects: int = 50, seed: Optional[int] = None) -> pd.DataFrame:
        if seed is None:
            seed = int(datetime.now().timestamp()) % 10000
        np.random.seed(seed)
        data = []
        
        for i in range(n_projects):
            project_id = f"VCS-{2020 + (i % 5)}-{1000+i}"
            project_type = np.random.choice(Config.PROJECT_TYPES)
            region = np.random.choice(list(Config.REGIONS_COORDS.keys()))
            lat, lon, location_name = Config.REGIONS_COORDS[region][np.random.randint(0, len(Config.REGIONS_COORDS[region]))]
            
            lat += np.random.uniform(-0.01, 0.01)
            lon += np.random.uniform(-0.01, 0.01)
            
            baseline_emissions = np.random.randint(5000, 50000)
            reduction_rate = np.random.uniform(0.15, 0.85)
            credits_generated = int(baseline_emissions * reduction_rate)
            
            buffer_percentage = np.random.uniform(0.10, 0.20)
            buffer_credits = int(credits_generated * buffer_percentage)
            net_credits = credits_generated - buffer_credits
            
            permanence_risk = np.random.choice(["Low", "Medium", "High"], p=[0.4, 0.4, 0.2])
            additionality_score = np.random.uniform(60, 95)
            verification_status = np.random.choice(Config.VERIFICATION_STATUSES, p=[0.5, 0.2, 0.2, 0.1])
            compliance_standard = np.random.choice(Config.COMPLIANCE_STANDARDS)
            
            start_date = datetime.now() - timedelta(days=np.random.randint(365, 365*5))
            vintage_year = start_date.year + np.random.randint(1, 10)
            
            data.append([
                project_id, project_type, region, location_name,
                credits_generated, net_credits, buffer_credits, baseline_emissions,
                np.random.randint(1, 10), verification_status, compliance_standard,
                permanence_risk, additionality_score, vintage_year,
                start_date, datetime.now(), lat, lon
            ])
        
        columns = [
            "ProjectID", "Type", "Region", "Location", 
            "TotalCredits", "NetCredits", "BufferCredits", "BaselineEmissions",
            "MonitoringPeriod", "VerificationStatus", "ComplianceStandard",
            "PermanenceRisk", "AdditionalityScore", "VintageYear",
            "StartDate", "LastUpdate", "Latitude", "Longitude"
        ]
        return pd.DataFrame(data, columns=columns)

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        issues = []
        if df.empty:
            issues.append("DataFrame is empty.")
        if not df['NetCredits'].ge(0).all():
            issues.append("Negative NetCredits found.")
        if not df['AdditionalityScore'].between(0, 100).all():
            issues.append("AdditionalityScore outside [0,100].")
        if not (df['BufferCredits'] <= df['TotalCredits']).all():
            issues.append("BufferCredits exceed TotalCredits.")
        if df[['Latitude', 'Longitude']].isnull().any().any():
            issues.append("Missing geographic coordinates.")
        return {"valid": len(issues) == 0, "issues": issues}


# ------------------------------
# Filtering Helper
# ------------------------------
def apply_filters(
    df: pd.DataFrame,
    regions: list,
    types: list,
    statuses: list,
    risks: list,
    credit_min: int,
    credit_max: int
) -> pd.DataFrame:
    filtered = df.copy()
    if regions:
        filtered = filtered[filtered["Region"].isin(regions)]
    if types:
        filtered = filtered[filtered["Type"].isin(types)]
    if statuses:
        filtered = filtered[filtered["VerificationStatus"].isin(statuses)]
    if risks:
        filtered = filtered[filtered["PermanenceRisk"].isin(risks)]
    filtered = filtered[filtered["NetCredits"].between(credit_min, credit_max)]
    return filtered


# ------------------------------
# Visualization Manager
# ------------------------------
class VisualizationManager:
    @staticmethod
    def _apply_branding(fig):
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=12, color="#333333"),
            title=dict(font=dict(size=16, color=Config.PRIMARY_COLOR)),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode="closest"
        )
        return fig

    @staticmethod
    def credits_by_type(df):
        if df.empty:
            return go.Figure().add_annotation(text="No data", x=0.5, y=0.5, showarrow=False, font_size=16)
        
        fig = px.bar(
            df,
            x="Type",
            y="NetCredits",
            color="Type",
            facet_col="Region",
            facet_col_wrap=2,
            title="Net Credits by Project Type & Region",
            labels={"NetCredits": "Net Credits (tCO₂e)"},
            color_discrete_map=Config.TYPE_COLORS,
            height=700
        )
        
        fig.update_traces(
            marker_line_width=1.2,
            marker_line_color="white",
            hovertemplate="<b>%{x}</b><br>Net Credits: %{y:,.0f} tCO₂e<extra></extra>"
        )
        
        fig.update_xaxes(tickangle=15, tickfont=dict(size=11))
        fig.for_each_annotation(lambda a: a.update(text=""))
        
        regions = df["Region"].unique()
        for i, region in enumerate(regions):
            fig.layout.annotations[i].update(
                text=region,
                font=dict(size=14, color=Config.PRIMARY_COLOR),
                yanchor="bottom",
                y=1.02
            )
        
        fig.update_layout(
            title_x=0.5,
            title_font_size=18,
            margin=dict(t=80, b=100, l=60, r=40)
        )
        
        return VisualizationManager._apply_branding(fig)

    @staticmethod
    def compliance_pie(df):
        if df.empty:
            return go.Figure().add_annotation(text="No data", x=0.5, y=0.5, showarrow=False, font_size=16)
        compliance_order = ["Verra VCS", "Gold Standard", "Climate Action Reserve", "American Carbon Registry"]
        color_sequence = ["#2E8B57", "#4682B4", "#5F9EA0", "#6495ED"]
        color_map = dict(zip(compliance_order, color_sequence))
        fig = px.pie(
            df,
            names="ComplianceStandard",
            title="Compliance Standards Distribution",
            hole=0.4,
            color="ComplianceStandard",
            color_discrete_map=color_map
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hoverinfo='label+percent+value',
            marker=dict(line=dict(color='#FFFFFF', width=2))
        )
        return VisualizationManager._apply_branding(fig)

    @staticmethod
    def risk_scatter(df, max_points=500):
        if df.empty:
            return go.Figure().add_annotation(text="No data", x=0.5, y=0.5, showarrow=False, font_size=16)
        df_plot = df.copy()
        if len(df_plot) > max_points:
            df_plot = df_plot.sample(n=max_points, random_state=42)
        df_plot['PermanenceRisk'] = pd.Categorical(df_plot['PermanenceRisk'], categories=["Low", "Medium", "High"], ordered=True)
        risk_colors = {"Low": "#28a745", "Medium": "#ffc107", "High": "#dc3545"}
        fig = px.scatter(
            df_plot,
            x="AdditionalityScore",
            y="NetCredits",
            color="PermanenceRisk",
            size="TotalCredits",
            hover_data=["ProjectID", "Type", "Region"],
            title="Risk Profile: Additionality vs. Net Credits",
            labels={
                "AdditionalityScore": "Additionality Score (%)",
                "NetCredits": "Net Credits (tCO₂e)"
            },
            color_discrete_map=risk_colors
        )
        fig.update_traces(marker=dict(opacity=0.85, line=dict(width=0.8, color='white')))
        return VisualizationManager._apply_branding(fig)

    @staticmethod
    def create_folium_popup(row) -> str:
        return f"""
        <div style="font-family: Arial; font-size: 12px;">
        <b>{row['ProjectID']}</b><br>
        Type: {row['Type']}<br>
        Location: {row['Location']}<br>
        Net Credits: {row['NetCredits']:,.0f} tCO₂e<br>
        Vintage: {row['VintageYear']}<br>
        Status: {row['VerificationStatus']}<br>
        Risk: {row['PermanenceRisk']}<br>
        Additionality: {row['AdditionalityScore']:.1f}%
        </div>
        """


# ------------------------------
# Report Generator (FIXED: alignment error resolved)
# ------------------------------
class ReportGenerator:
    @staticmethod
    def fig_to_png_bytes(fig) -> bytes:
        buf = io.BytesIO()
        try:
            fig.write_image(buf, format="png", width=800, height=500, scale=2, engine="kaleido")
        except Exception as e:
            logger.warning(f"Kaleido export failed; using fallback: {e}")
            try:
                fig.write_image(buf, format="png", width=800, height=500, scale=1)
            except Exception as e2:
                logger.error(f"Fallback export also failed: {e2}")
                img = PILImage.new('RGB', (800, 500), color=(245, 245, 245))
                draw = ImageDraw.Draw(img)
                draw.text((20, 240), "Chart not available. Install kaleido.", fill=(100, 100, 100))
                img.save(buf, format="PNG")
        buf.seek(0)
        return buf.read()

    @staticmethod
    def _create_legend_table():
        data = [["Project Type", ""]]
        col_widths = [2.8 * inch, 0.6 * inch]
        style = [
            ('BACKGROUND', (0, 0), (0, 0), rl_colors.Color(0.18, 0.55, 0.34)),
            ('TEXTCOLOR', (0, 0), (0, 0), rl_colors.white),
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, 0), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]
        for ptype in Config.PROJECT_TYPES:
            data.append([ptype, ""])
            hex_color = Config.TYPE_COLORS.get(ptype, "#999999")
            r = int(hex_color[1:3], 16) / 255.0
            g = int(hex_color[3:5], 16) / 255.0
            b = int(hex_color[5:7], 16) / 255.0
            row_idx = len(data) - 1
            style.append(('BACKGROUND', (1, row_idx), (1, row_idx), rl_colors.Color(r, g, b)))
            style.append(('LINEABOVE', (0, row_idx), (-1, row_idx), 0.5, rl_colors.grey))
        table = Table(data, colWidths=col_widths, rowHeights=20)
        table.setStyle(TableStyle(style))
        return table

    @staticmethod
    def generate_pdf(df: pd.DataFrame, figures: Dict[str, go.Figure]) -> Optional[bytes]:
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
            
            # Title style (centered)
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                textColor=rl_colors.Color(0.18, 0.55, 0.34),
                spaceAfter=10,
                alignment=1  # 0=left, 1=center, 2=right
            )
            
            # Centered normal text style
            centered_style = ParagraphStyle(
                'Centered',
                parent=styles['Normal'],
                alignment=1,
                fontSize=11,
                leading=14
            )
            
            normal_style = styles['Normal']
            normal_style.fontSize = 11
            normal_style.leading = 14

            story = []

            # Logo
            try:
                logo = Image("assets/greentrust_logo.png", width=2.2 * inch, height=0.7 * inch)
                story.append(logo)
                story.append(Spacer(1, 8))
            except:
                pass

            story.append(Paragraph("GreenTrust Insights", title_style))
            story.append(Paragraph("<b>Smart AI. Trusted Climate Impact.</b>", centered_style))  # ✅ FIXED HERE
            story.append(Spacer(1, 20))

            # Executive Summary
            story.append(Paragraph("<b>Executive Summary</b>", styles['Heading1']))
            summary_data = [
                ["Total Active Projects:", f"{len(df):,}"],
                ["Total Net Credits Generated:", f"{df['NetCredits'].sum():,.0f} tCO₂e"],
                ["Average Additionality Score:", f"{df['AdditionalityScore'].mean():.1f}%"],
                ["Verified Projects:", f"{len(df[df['VerificationStatus']=='Verified'])} ({(len(df[df['VerificationStatus']=='Verified'])/len(df)*100):.1f}%)"],
                ["Portfolio Vintage Range:", f"{df['VintageYear'].min()} – {df['VintageYear'].max()}"]
            ]
            summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,0), (-1,-1), 11),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 18))

            # Legend
            story.append(Paragraph("<b>Project Type Legend</b>", styles['Heading2']))
            story.append(ReportGenerator._create_legend_table())
            story.append(Spacer(1, 22))

            # Charts
            chart_info = [
                ("Net Credits by Project Type & Region", figures.get("Net Credits by Project Type & Region")),
                ("Compliance Standards Distribution", figures.get("Compliance Standards Distribution")),
                ("Risk Profile: Additionality vs. Net Credits", figures.get("Risk Profile: Additionality vs. Net Credits"))
            ]
            for title, fig in chart_info:
                if fig is not None:
                    story.append(Paragraph(f"<b>{title}</b>", styles['Heading2']))
                    img_bytes = ReportGenerator.fig_to_png_bytes(fig)
                    img = Image(io.BytesIO(img_bytes), width=6.6 * inch, height=4.2 * inch)
                    story.append(img)
                    story.append(Spacer(1, 20))

            # Top Projects
            story.append(Paragraph("<b>Top 10 Projects by Net Credits</b>", styles['Heading1']))
            if not df.empty:
                top = df.nlargest(10, 'NetCredits')[['ProjectID', 'Type', 'NetCredits', 'VerificationStatus']]
                table_data = [['Project ID', 'Type', 'Net Credits (tCO₂e)', 'Status']]
                for _, row in top.iterrows():
                    table_data.append([str(row['ProjectID']), str(row['Type']), f"{row['NetCredits']:,.0f}", str(row['VerificationStatus'])])
                table = Table(table_data, colWidths=[1.6*inch, 1.4*inch, 1.5*inch, 1.3*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), rl_colors.Color(0.18, 0.55, 0.34)),
                    ('TEXTCOLOR', (0,0), (-1,0), rl_colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,0), 11),
                    ('BOTTOMPADDING', (0,0), (-1,0), 8),
                    ('BACKGROUND', (0,1), (-1,-1), rl_colors.Color(0.97, 0.98, 0.97)),
                    ('GRID', (0,0), (-1,-1), 0.8, rl_colors.Color(0.85, 0.85, 0.85)),
                    ('FONTSIZE', (0,1), (-1,-1), 10),
                ]))
                story.append(table)

            story.append(Spacer(1, 24))
            story.append(Paragraph("<i>© GreenTrust Insights — Climate Intelligence Platform</i>", centered_style))  # ✅ FIXED HERE

            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
        except Exception as e:
            logger.exception("PDF generation failed")
            return None


# ------------------------------
# Main App
# ------------------------------
def main():
    st.set_page_config(
        page_title="GreenTrust Insights",
        page_icon="assets/greentrust_favicon.png",
        layout="wide"
    )
    
    st.title("🌍 GreenTrust Insights")
    st.markdown(f"**{Config.TAGLINE}**")
    st.markdown("---")

    st.sidebar.header("⚙️ Data Settings")
    fixed_dataset = st.sidebar.checkbox("📌 Keep dataset fixed", value=True)
    seed = 42 if fixed_dataset else None
    refresh = st.sidebar.button("🔄 Refresh dataset")
    if refresh:
        st.session_state.pop('df', None)

    if 'df' not in st.session_state:
        with st.spinner("Loading portfolio data..."):
            st.session_state.df = CarbonProjectDataManager.generate_realistic_carbon_projects(seed=seed)
    df = st.session_state.df

    validation = CarbonProjectDataManager.validate_dataframe(df)
    if not validation["valid"]:
        st.error("Data validation issues: " + "; ".join(validation["issues"]))
        return

    st.sidebar.header("🔍 Filters")
    region_filter = st.sidebar.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    type_filter = st.sidebar.multiselect("Project Type", sorted(df["Type"].unique()), default=sorted(df["Type"].unique()))
    status_filter = st.sidebar.multiselect("Verification Status", sorted(df["VerificationStatus"].unique()), default=sorted(df["VerificationStatus"].unique()))
    risk_filter = st.sidebar.multiselect("Permanence Risk", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
    
    min_credits = int(df["NetCredits"].min())
    max_credits = int(df["NetCredits"].max())
    credit_range = st.sidebar.slider("Net Credits (tCO₂e)", min_credits, max_credits, (min_credits, max_credits))
    
    if st.sidebar.button("↺ Reset Filters"):
        st.rerun()

    filtered_df = apply_filters(df, region_filter, type_filter, status_filter, risk_filter, credit_range[0], credit_range[1])
    if filtered_df.empty:
        st.warning("⚠️ No projects match your filters.")
        return

    fig1 = VisualizationManager.credits_by_type(filtered_df)
    fig2 = VisualizationManager.compliance_pie(filtered_df)
    fig3 = VisualizationManager.risk_scatter(filtered_df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📈 Analytics", "🗺️ Map", "⚖️ Compare", "📋 Report"])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Projects", len(filtered_df))
        col2.metric("Net Credits", f"{filtered_df['NetCredits'].sum():,.0f} tCO₂e")
        col3.metric("Avg Additionality", f"{filtered_df['AdditionalityScore'].mean():.1f}%")
        verified = len(filtered_df[filtered_df['VerificationStatus'] == 'Verified'])
        rate = (verified / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
        col4.metric("Verification Rate", f"{rate:.1f}%")

    with tab2:
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(fig1, use_container_width=True)
        with col2: st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        if not filtered_df.empty:
            m = folium.Map(
                location=[filtered_df['Latitude'].mean(), filtered_df['Longitude'].mean()],
                zoom_start=3,
                tiles="CartoDB positron"
            )
            marker_cluster = MarkerCluster().add_to(m)
            for _, row in filtered_df.iterrows():
                color = "green" if row["VerificationStatus"] == "Verified" else "orange"
                popup_html = VisualizationManager.create_folium_popup(row)
                folium.Marker(
                    [row["Latitude"], row["Longitude"]],
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=f"{row['ProjectID']} ({row['Type']})",
                    icon=folium.Icon(color=color, icon="leaf", prefix="fa")
                ).add_to(marker_cluster)
            st_folium(m, width="100%", height=500)
        else:
            st.info("No geographic data to display.")

    with tab4:
        project_ids = sorted(filtered_df["ProjectID"].unique())
        if len(project_ids) >= 2:
            col1, col2 = st.columns(2)
            p1 = col1.selectbox("Project 1", project_ids, key="p1")
            p2 = col2.selectbox("Project 2", project_ids, key="p2")
            if p1 != p2:
                d1 = filtered_df[filtered_df["ProjectID"] == p1].iloc[0]
                d2 = filtered_df[filtered_df["ProjectID"] == p2].iloc[0]
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"#### {d1['ProjectID']}")
                    st.write(f"**Type:** {d1['Type']}")
                    st.metric("Net Credits", f"{d1['NetCredits']:,.0f}")
                    st.metric("Additionality", f"{d1['AdditionalityScore']:.1f}%")
                    st.write(f"**Risk:** {d1['PermanenceRisk']}")
                with col2:
                    st.markdown(f"#### {d2['ProjectID']}")
                    st.write(f"**Type:** {d2['Type']}")
                    st.metric("Net Credits", f"{d2['NetCredits']:,.0f}")
                    st.metric("Additionality", f"{d2['AdditionalityScore']:.1f}%")
                    st.write(f"**Risk:** {d2['PermanenceRisk']}")
                comp_df = pd.DataFrame({
                    'Project': [p1, p2],
                    'Net Credits': [d1['NetCredits'], d2['NetCredits']],
                    'Additionality': [d1['AdditionalityScore'], d2['AdditionalityScore']]
                })
                fig = px.bar(comp_df, x='Project', y=['Net Credits', 'Additionality'], barmode='group', title="Project Comparison")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select 2+ projects to compare.")

    with tab5:
        st.write("Generate a branded portfolio report with visuals and a color legend.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Generate PDF Report", type="primary"):
                with st.spinner("Generating branded report..."):
                    figures = {
                        "Net Credits by Project Type & Region": fig1,
                        "Compliance Standards Distribution": fig2,
                        "Risk Profile: Additionality vs. Net Credits": fig3
                    }
                    pdf = ReportGenerator.generate_pdf(filtered_df, figures)
                    if pdf:
                        st.download_button(
                            "📥 Download PDF",
                            data=pdf,
                            file_name=f"GreenTrust_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.error("Failed to generate report. Check logs.")
        with col2:
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("📤 Export CSV", csv, "carbon_projects.csv", "text/csv")

    st.markdown(
        f"---\n<small>**Data**: {len(filtered_df)} of {len(df)} projects • Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</small>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()