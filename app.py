import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import datetime
import random

# 📂 Load data and clean columns
entities = pd.read_csv("data/offshore_entities.csv")
officers_df = pd.read_csv("data/officers.csv")
ownership_df = pd.read_csv("data/ownership_edges.csv")
ownership_df.columns = ownership_df.columns.str.strip()  # <-- add this to remove extra spaces

# 🧼 Strip whitespace from column names
entities.columns = entities.columns.str.strip()
officers_df.columns = officers_df.columns.str.strip()

# Set Streamlit page config
st.set_page_config(page_title="ShadowFinance Analyzer", layout="wide")

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "ShadowFinance Analyzer",
        ["📊 Network Graph", "📈 Incorporation Timeline", "📄 Suspicion Table", "🔁 Ownership Loop Detection", "🌐 Geolocation Map", "💸 Money Flow Simulation"],
        icons=["graph-up", "calendar-event", "table", "arrow-repeat", "geo-alt", "currency-exchange"],
        menu_icon="cash-coin",
        default_index=0,
    )

# Page 1: Network Graph
if selected == "📊 Network Graph":
    st.title("💰 Black Money Shell Network Analyzer")
    st.write("### 🧘️ Shell Company Network Graph")

    threshold = st.slider("Set Suspicion Score Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    if st.button("🔄 Regenerate Suspicion Scores"):
        entities['suspicion_score'] = np.random.rand(len(entities))
    else:
        if 'suspicion_score' not in entities.columns:
            entities['suspicion_score'] = np.random.rand(len(entities))

    G = nx.Graph()
    for _, row in entities.iterrows():
        if row['suspicion_score'] >= threshold:
            G.add_node(row['entity_id'], label=row['entity_name'], type='entity', suspicion=row['suspicion_score'])

    for _, row in officers_df.iterrows():
        if row['linked_entity_id'] in G.nodes:
            G.add_node(row['officer_id'], label=row['officer_name'], type='officer')
            G.add_edge(row['officer_id'], row['linked_entity_id'], role=row['role'])

    node_colors = []
    for node, data in G.nodes(data=True):
        if data['type'] == 'entity':
            score = data.get('suspicion', 0.0)
            node_colors.append((1, 1 - score, 1 - score))
        else:
            node_colors.append('orange')

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 7))
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=1000, edge_color='gray')
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=9)
    st.pyplot(plt)

    st.markdown("""
    **Legend:**
    - 🔵 SkyBlue to 🔴 Red = Companies (Red = More suspicious)
    - 🟠 Orange = People (officers)
    - ⚙️ Edges = Links between people and companies
    """)

# Page 2: Incorporation Timeline
elif selected == "📈 Incorporation Timeline":
    st.title("🗖️ Incorporation Timeline Analysis")

    if 'incorporation_date' not in entities.columns:
        st.warning("No incorporation date column found in the dataset.")
    else:
        entities['incorporation_date'] = pd.to_datetime(entities['incorporation_date'], errors='coerce')
        df = entities.dropna(subset=['incorporation_date'])
        df['year'] = df['incorporation_date'].dt.year

        year_counts = df['year'].value_counts().sort_index().reset_index()
        year_counts.columns = ['Year', 'Number of Companies']

        fig = px.line(year_counts, x='Year', y='Number of Companies', markers=True,
                      title="📈 Number of Shell Companies Incorporated Over Time")
        st.plotly_chart(fig, use_container_width=True)

# Page 3: Suspicion Table
elif selected == "📄 Suspicion Table":
    st.title("📈 Suspicion Scores Table")
    if 'suspicion_score' not in entities.columns:
        entities['suspicion_score'] = np.random.rand(len(entities))
    st.dataframe(
        entities[['entity_name', 'suspicion_score']]
        .sort_values(by='suspicion_score', ascending=False)
        .reset_index(drop=True)
    )

# Page 4: Ownership Loop Detection
elif selected == "🔁 Ownership Loop Detection":
    st.title("🔍 Ownership Loop Detection")
    st.write("Detecting circular ownership loops in shell company networks...")

    DG = nx.DiGraph()
    for _, row in entities.iterrows():
        DG.add_node(row['entity_id'], label=row['entity_name'])

    for _, row in officers_df.iterrows():
        if row['linked_entity_id'] in DG.nodes:
            DG.add_edge(row['officer_id'], row['linked_entity_id'])

    for officer in officers_df['officer_id'].unique():
        linked_entities = officers_df[officers_df['officer_id'] == officer]['linked_entity_id'].tolist()
        for i in range(len(linked_entities)):
            for j in range(len(linked_entities)):
                if i != j:
                    DG.add_edge(linked_entities[i], linked_entities[j])

    loops = list(nx.simple_cycles(DG))
    suspicious_loops = [loop for loop in loops if len(loop) > 2 and all(node in entities['entity_id'].values for node in loop)]

    if suspicious_loops:
        st.success(f"🔴 {len(suspicious_loops)} suspicious ownership loops found!")
        for i, loop in enumerate(suspicious_loops, 1):
            loop_names = [DG.nodes[n].get('label', n) for n in loop]
            st.markdown(f"**Loop {i}:** {' → '.join(loop_names)}")

        loop_graph = nx.DiGraph()
        for loop in suspicious_loops:
            for i in range(len(loop)):
                src = loop[i]
                tgt = loop[(i + 1) % len(loop)]
                loop_graph.add_edge(src, tgt)
                loop_graph.nodes[src]['label'] = DG.nodes[src].get('label', src)
                loop_graph.nodes[tgt]['label'] = DG.nodes[tgt].get('label', tgt)

        pos = nx.spring_layout(loop_graph, seed=42)
        plt.figure(figsize=(12, 6))
        nx.draw_networkx(loop_graph, pos, node_color='red', edge_color='black', with_labels=False, node_size=1000)
        labels = nx.get_node_attributes(loop_graph, 'label')
        nx.draw_networkx_labels(loop_graph, pos, labels, font_size=9, font_color='white')
        st.pyplot(plt)
    else:
        st.info("✅ No suspicious circular ownership loops detected.")

# Page 5: Enhanced Geolocation Map
elif selected == "🌐 Geolocation Map":
    st.title("🌍 Shell Company Geolocation Map")
    country_coords = {
        "Panama": (8.538, -80.782),
        "British Virgin Islands": (18.4207, -64.64),
        "Singapore": (1.3521, 103.8198),
        "UAE": (23.4241, 53.8478),
        "Cyprus": (35.1264, 33.4299),
        "Luxembourg": (49.8153, 6.1296),
        "Cayman Islands": (19.3133, -81.2546)
    }
    if 'country' in entities.columns:
        geo_df = entities[entities['country'].isin(country_coords.keys())].copy()
        geo_df['lat'] = geo_df['country'].apply(lambda x: country_coords[x][0])
        geo_df['lon'] = geo_df['country'].apply(lambda x: country_coords[x][1])

        fig = px.scatter_geo(
            geo_df,
            lat='lat',
            lon='lon',
            text='entity_name',
            color='suspicion_score',
            color_continuous_scale='Reds',
            hover_name='entity_name',
            size='suspicion_score',
            projection="natural earth",
            title="🌐 Shell Entity Locations by Suspicion Level"
        )
        fig.update_layout(
            geo=dict(
                showland=True,
                landcolor="rgb(243, 243, 243)",
                coastlinecolor="gray",
                showocean=True,
                oceancolor="LightBlue"
            ),
            margin={"r":0,"t":40,"l":0,"b":0},
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No 'country' column found in your dataset. Cannot plot geolocation map.")

# Page 6: Money Flow Simulation
elif selected == "💸 Money Flow Simulation":
    st.title("💸 Money Flow Simulation")

    def build_ownership_graph(df):
        G = nx.DiGraph()
        for _, row in df.iterrows():
            G.add_edge(row['owner_entity_id'], row['owned_entity_id'], weight=row['ownership_percent'])
        return G

    def simulate_money_flow(graph, start_entity, max_depth=5):
        paths = []
        def dfs(node, path, depth):
            if depth > max_depth:
                return
            for neighbor in graph.successors(node):
                new_path = path + [neighbor]
                paths.append(new_path)
                dfs(neighbor, new_path, depth + 1)
        dfs(start_entity, [start_entity], 0)
        return paths

    def visualize_money_flow(graph, paths):
        fig = go.Figure()
        colors = ['blue', 'green', 'purple', 'orange', 'red']
        for i, path in enumerate(paths[:10]):
            x, y = [], []
            for _ in path:
                x.append(random.random() * 10)
                y.append(random.random() * 10)
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines+markers+text',
                name=f'Path {i+1}',
                text=path,
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        fig.update_layout(title="💸 Simulated Money Flow Paths", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    ownership_graph = build_ownership_graph(ownership_df)
    all_entities = list(ownership_graph.nodes)
    selected_entity = st.selectbox("Select an Entity to Simulate Flow From", all_entities)
    max_depth = st.slider("Select Max Depth of Flow", 1, 10, 5)
    st.info("Tracing flow paths up to given depth...")

    money_paths = simulate_money_flow(ownership_graph, selected_entity, max_depth)

    if money_paths:
        st.success(f"Found {len(money_paths)} simulated paths of money flow.")
        visualize_money_flow(ownership_graph, money_paths)
    else:
        st.warning("No money flow paths found from the selected entity.")

# 📂 7. Entity Details Sidebar (Append this at the end of app.py)

# Divider in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🧾 Entity Detail Viewer")

# Make sure entities and officers_df exist
try:
    entity_ids = entities['entity_id'].unique()
    selected_node = st.sidebar.selectbox("🔍 Select an Entity ID", entity_ids)

    # Get entity info
    entity_info = entities[entities['entity_id'] == selected_node].iloc[0]

    # 🧾 Display Entity Details
    st.sidebar.markdown("### 🧾 Entity Information")
    st.sidebar.markdown(f"- **Entity ID:** {entity_info['entity_id']}")
    st.sidebar.markdown(f"- **Name:** {entity_info['entity_name']}")
    st.sidebar.markdown(f"- **Suspicion Score:** {round(entity_info['suspicion_score'], 2)}")
    st.sidebar.markdown(f"- **Incorporation Date:** {entity_info['incorporation_date']}")
    st.sidebar.markdown(f"- **Country:** {entity_info['country']}")

    # 👥 Connected Officers
    st.sidebar.markdown("### 👤 Connected Officers")
    connected_officers = officers_df[officers_df['linked_entity_id'] == selected_node]

    if not connected_officers.empty:
        for _, officer in connected_officers.iterrows():
            officer_name = officer.get("officer_name", "Unknown")
            role = officer.get("role", "Unknown")
            st.sidebar.markdown(f"- {officer_name} ({role})")
    else:
        st.sidebar.markdown("- No connected officers found.")

except Exception as e:
    st.sidebar.error("⚠️ Error loading entity info.")
    st.sidebar.code(str(e))
