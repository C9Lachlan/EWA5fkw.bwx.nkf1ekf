import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime
try:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    st.warning("NetworkX library not found. Keyword Groups tab will be disabled. Install with: pip install networkx")

# Page configuration
st.set_page_config(
    page_title="SEO Keyword Research Tool",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        border: none;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("SEO Keyword Research Tool")
st.markdown("Analyze keyword cannibalization and competitor data to optimize your SEO strategy.")

# Initialize session state
if 'cannibalization_data' not in st.session_state:
    st.session_state['cannibalization_data'] = None
if 'ahrefs_data' not in st.session_state:
    st.session_state['ahrefs_data'] = None

@st.cache_data
def cached_process_keyword_groups(df_hash, df):
    """Cached version of keyword groups processing."""
    return process_keyword_groups(df)

# Helper Functions
def validate_cannibalization_csv(df):
    """Validate cannibalization export CSV has required columns."""
    required_cols = ['Query', 'Landing Page', 'Impressions', 'Url Clicks', 'URL CTR', 'Average Position']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    # Clean column names (handle spaces and special characters)
    df.columns = df.columns.str.strip()
    
    # Data type conversions with error handling
    try:
        df['Impressions'] = pd.to_numeric(df['Impressions'], errors='coerce').fillna(0).astype(int)
        df['Url Clicks'] = pd.to_numeric(df['Url Clicks'], errors='coerce').fillna(0).astype(int)
        df['URL CTR'] = pd.to_numeric(df['URL CTR'], errors='coerce').fillna(0)
        df['Average Position'] = pd.to_numeric(df['Average Position'], errors='coerce').fillna(0)
    except Exception as e:
        return False, f"Error converting data types: {str(e)}"
    
    return True, df

def validate_ahrefs_csv(df):
    """Validate Ahrefs competitor data CSV has required columns."""
    required_cols = ['Keyword', 'Volume', 'KD', 'CPC']
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    # Data type conversions
    try:
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).astype(int)
        df['KD'] = pd.to_numeric(df['KD'], errors='coerce').fillna(0)
        df['CPC'] = pd.to_numeric(df['CPC'], errors='coerce').fillna(0)
    except Exception as e:
        return False, f"Error converting data types: {str(e)}"
    
    return True, df

def process_current_keywords(df):
    """Process cannibalization data to create Current Keywords table."""
    # Group by Query and aggregate
    grouped = df.groupby('Query').agg({
        'Url Clicks': 'sum',
        'Impressions': 'sum',
        'Landing Page': 'nunique'  # Count unique landing pages
    }).reset_index()
    
    # Rename the landing page count column
    grouped.rename(columns={'Landing Page': 'Landing Pages'}, inplace=True)
    
    # Calculate cannibalization percentage
    # For each query, find the max impressions for a single landing page
    max_impressions = df.groupby('Query').apply(
        lambda x: x.groupby('Landing Page')['Impressions'].sum().max()
    ).reset_index(name='Max_LP_Impressions')
    
    # Merge the data
    result = grouped.merge(max_impressions, on='Query')
    
    # Calculate cannibalization percentage
    result['Cannibalisation %'] = np.where(
        result['Impressions'] > 0,
        ((result['Impressions'] - result['Max_LP_Impressions']) / result['Impressions'] * 100).round(2),
        0
    )
    
    # Rename and reorder columns for final output
    result = result[['Query', 'Url Clicks', 'Impressions', 'Landing Pages', 'Cannibalisation %']]
    result.columns = ['Query', 'Clicks', 'Impressions', 'Landing Pages', 'Cannibalisation %']
    
    # Sort by impressions descending
    result = result.sort_values('Impressions', ascending=False)
    
    return result

def process_keyword_details(df, selected_query):
    """Process keyword details for a specific query."""
    # Filter data for selected query
    filtered = df[df['Query'] == selected_query].copy()
    
    # Calculate total impressions for the query
    total_impressions = filtered['Impressions'].sum()
    
    # Calculate impression percentage
    filtered['Impression %'] = np.where(
        total_impressions > 0,
        (filtered['Impressions'] / total_impressions * 100).round(2),
        0
    )
    
    # Select and rename columns
    result = filtered[['Landing Page', 'Url Clicks', 'Impressions', 'Impression %', 'URL CTR', 'Average Position']]
    result.columns = ['Landing Page', 'Clicks', 'Impressions', 'Impression %', 'CTR', 'Average Position']
    
    # Convert CTR to percentage if it's not already
    if result['CTR'].max() <= 1:
        result['CTR'] = (result['CTR'] * 100).round(2)
    
    # Sort by impressions descending
    result = result.sort_values('Impressions', ascending=False)
    
    return result

def process_current_pages(df):
    """Process cannibalization data to create Current Pages table."""
    # Group by Landing Page and aggregate
    grouped = df.groupby('Landing Page').agg({
        'Url Clicks': 'sum',
        'Impressions': 'sum',
        'Query': 'nunique'  # Count unique queries
    }).reset_index()
    
    # Rename columns for final output
    grouped.columns = ['Landing Page', 'Clicks', 'Impressions', 'Queries']
    
    # Sort by impressions descending
    grouped = grouped.sort_values('Impressions', ascending=False)
    
    return grouped

def process_page_details(df, selected_page):
    """Process page details for a specific landing page."""
    # Filter data for selected landing page
    filtered = df[df['Landing Page'] == selected_page].copy()
    
    # Select and rename columns
    result = filtered[['Query', 'Url Clicks', 'Impressions', 'URL CTR', 'Average Position']]
    result.columns = ['Query', 'Clicks', 'Impressions', 'CTR', 'Average Position']
    
    # Convert CTR to percentage if it's not already
    if result['CTR'].max() <= 1:
        result['CTR'] = (result['CTR'] * 100).round(2)
    
    # Sort by impressions descending
    result = result.sort_values('Impressions', ascending=False)
    
    return result

def process_competitor_keywords(df):
    """Process Ahrefs data to create Competitor Keywords table."""
    # Get all competitor domain columns (excluding the first one which is our site)
    traffic_cols = [col for col in df.columns if 'Traffic' in col]
    position_cols = [col for col in df.columns if 'Position' in col and 'Average' not in col]
    url_cols = [col for col in df.columns if 'URL' in col]
    
    # Skip the first domain (our site)
    if len(traffic_cols) > 1:
        competitor_traffic_cols = traffic_cols[1:]
        competitor_position_cols = position_cols[1:]
        competitor_url_cols = url_cols[1:]
    else:
        # No competitor data available
        return pd.DataFrame(columns=['Keyword', 'Volume', 'KD', 'Competitor # Pages', 'Top Page', 'Traffic', 'Position'])
    
    result_rows = []
    
    for idx, row in df.iterrows():
        # Count competitor pages ranking for this keyword (non-zero traffic)
        competitor_count = 0
        max_traffic = 0
        top_page = ""
        top_position = 0
        
        for i, traffic_col in enumerate(competitor_traffic_cols):
            traffic_val = row[traffic_col]
            if pd.notna(traffic_val) and traffic_val > 0:
                competitor_count += 1
                if traffic_val > max_traffic:
                    max_traffic = traffic_val
                    top_page = row[competitor_url_cols[i]] if pd.notna(row[competitor_url_cols[i]]) else ""
                    top_position = row[competitor_position_cols[i]] if pd.notna(row[competitor_position_cols[i]]) else 0
        
        # Only include keywords where at least one competitor ranks
        if competitor_count > 0:
            result_rows.append({
                'Keyword': row['Keyword'],
                'Volume': row['Volume'],
                'KD': row['KD'],
                'Competitor # Pages': competitor_count,
                'Top Page': top_page,
                'Traffic': max_traffic,
                'Position': top_position
            })
    
    result = pd.DataFrame(result_rows)
    
    # Sort by volume descending
    if not result.empty:
        result = result.sort_values('Volume', ascending=False)
    
    return result

def process_competitor_keyword_details(df, selected_keyword):
    """Process competitor details for a specific keyword."""
    # Filter data for selected keyword
    keyword_row = df[df['Keyword'] == selected_keyword]
    
    if keyword_row.empty:
        return pd.DataFrame(columns=['URL', 'Traffic', 'Position'])
    
    keyword_row = keyword_row.iloc[0]
    
    # Get all competitor columns (excluding the first one which is our site)
    traffic_cols = [col for col in df.columns if 'Traffic' in col]
    position_cols = [col for col in df.columns if 'Position' in col and 'Average' not in col]
    url_cols = [col for col in df.columns if 'URL' in col]
    
    # Skip the first domain (our site)
    if len(traffic_cols) > 1:
        competitor_traffic_cols = traffic_cols[1:]
        competitor_position_cols = position_cols[1:]
        competitor_url_cols = url_cols[1:]
    else:
        return pd.DataFrame(columns=['URL', 'Traffic', 'Position'])
    
    result_rows = []
    
    for i, traffic_col in enumerate(competitor_traffic_cols):
        traffic_val = keyword_row[traffic_col]
        if pd.notna(traffic_val) and traffic_val > 0:
            url_val = keyword_row[competitor_url_cols[i]] if pd.notna(keyword_row[competitor_url_cols[i]]) else ""
            position_val = keyword_row[competitor_position_cols[i]] if pd.notna(keyword_row[competitor_position_cols[i]]) else 0
            
            result_rows.append({
                'URL': url_val,
                'Traffic': traffic_val,
                'Position': position_val
            })
    
    result = pd.DataFrame(result_rows)
    
    # Sort by traffic descending
    if not result.empty:
        result = result.sort_values('Traffic', ascending=False)
    
    return result

def process_competitor_pages(df):
    """Process Ahrefs data to create Competitor Pages table."""
    # Get all competitor domain columns (excluding the first one which is our site)
    traffic_cols = [col for col in df.columns if 'Traffic' in col]
    position_cols = [col for col in df.columns if 'Position' in col and 'Average' not in col]
    url_cols = [col for col in df.columns if 'URL' in col]
    
    # Skip the first domain (our site)
    if len(traffic_cols) > 1:
        competitor_traffic_cols = traffic_cols[1:]
        competitor_position_cols = position_cols[1:]
        competitor_url_cols = url_cols[1:]
    else:
        # No competitor data available
        return pd.DataFrame(columns=['URL', '# Keywords', 'Traffic', 'Volume'])
    
    # Dictionary to store aggregated data by URL
    url_data = {}
    
    for idx, row in df.iterrows():
        for i, url_col in enumerate(competitor_url_cols):
            url = row[url_col]
            traffic = row[competitor_traffic_cols[i]]
            
            # Only process if URL exists and has traffic
            if pd.notna(url) and url and pd.notna(traffic) and traffic > 0:
                if url not in url_data:
                    url_data[url] = {
                        'keywords': 0,
                        'traffic': 0,
                        'volume': 0
                    }
                
                url_data[url]['keywords'] += 1
                url_data[url]['traffic'] += traffic
                url_data[url]['volume'] += row['Volume']
    
    # Convert to DataFrame
    result_rows = []
    for url, data in url_data.items():
        result_rows.append({
            'URL': url,
            '# Keywords': data['keywords'],
            'Traffic': data['traffic'],
            'Volume': data['volume']
        })
    
    result = pd.DataFrame(result_rows)
    
    # Sort by traffic descending
    if not result.empty:
        result = result.sort_values('Traffic', ascending=False)
    
    return result

def process_competitor_page_details(df, selected_url):
    """Process competitor page details for a specific URL."""
    # Get all competitor columns (excluding the first one which is our site)
    traffic_cols = [col for col in df.columns if 'Traffic' in col]
    position_cols = [col for col in df.columns if 'Position' in col and 'Average' not in col]
    url_cols = [col for col in df.columns if 'URL' in col]
    
    # Skip the first domain (our site)
    if len(traffic_cols) > 1:
        competitor_traffic_cols = traffic_cols[1:]
        competitor_position_cols = position_cols[1:]
        competitor_url_cols = url_cols[1:]
    else:
        return pd.DataFrame(columns=['Keyword', 'Traffic', 'Volume', 'Position'])
    
    result_rows = []
    
    for idx, row in df.iterrows():
        for i, url_col in enumerate(competitor_url_cols):
            url = row[url_col]
            
            # Check if this URL matches the selected URL
            if pd.notna(url) and url == selected_url:
                traffic = row[competitor_traffic_cols[i]]
                position = row[competitor_position_cols[i]]
                
                if pd.notna(traffic) and traffic > 0:
                    result_rows.append({
                        'Keyword': row['Keyword'],
                        'Traffic': traffic,
                        'Volume': row['Volume'],
                        'Position': position
                    })
    
    result = pd.DataFrame(result_rows)
    
    # Sort by traffic descending
    if not result.empty:
        result = result.sort_values('Traffic', ascending=False)
    
    return result

def get_competitor_keywords_sorted(df):
    """Get keywords sorted by volume where competitors rank."""
    # Process to get only keywords where competitors rank
    competitor_df = process_competitor_keywords(df)
    if not competitor_df.empty:
        return competitor_df['Keyword'].tolist()
    return []

def process_keyword_groups(df):
    """Process Ahrefs data to create keyword groups using Louvain community detection."""
    # Get all URL columns (including our site and competitors)
    url_cols = [col for col in df.columns if 'URL' in col]
    traffic_cols = [col for col in df.columns if 'Traffic' in col]
    
    if len(url_cols) == 0:
        return pd.DataFrame(columns=['Name', 'Keywords', 'Total Volume', 'Top URL', 'Total Traffic'])
    
    # Build keyword co-occurrence matrix based on shared URLs
    keywords = df['Keyword'].tolist()
    n_keywords = len(keywords)
    
    # Create co-occurrence matrix
    cooccurrence_matrix = np.zeros((n_keywords, n_keywords))
    
    for i in range(n_keywords):
        for j in range(i+1, n_keywords):
            shared_urls = 0
            # Check how many URLs rank for both keywords
            for url_col in url_cols:
                url_i = df.iloc[i][url_col]
                url_j = df.iloc[j][url_col]
                # If both keywords have the same URL ranking for them
                if pd.notna(url_i) and pd.notna(url_j) and url_i == url_j and url_i != '':
                    shared_urls += 1
            
            cooccurrence_matrix[i][j] = shared_urls
            cooccurrence_matrix[j][i] = shared_urls
    
    # Create a graph from the co-occurrence matrix
    G = nx.Graph()
    
    # Add nodes (keywords)
    for i, keyword in enumerate(keywords):
        G.add_node(i, name=keyword, volume=df.iloc[i]['Volume'])
    
    # Add edges where keywords share URLs
    for i in range(n_keywords):
        for j in range(i+1, n_keywords):
            if cooccurrence_matrix[i][j] > 0:
                G.add_edge(i, j, weight=cooccurrence_matrix[i][j])
    
    # Apply Louvain community detection
    if G.number_of_edges() > 0:
        communities = louvain_communities(G, seed=42)
    else:
        # If no edges, each keyword is its own community
        communities = [{i} for i in range(n_keywords)]
    
    # Process communities into groups
    groups_data = []
    
    for community_nodes in communities:
        community_keywords = [keywords[node] for node in community_nodes]
        community_volumes = [df.iloc[node]['Volume'] for node in community_nodes]
        
        # Name the group after the highest volume keyword
        max_volume_idx = community_volumes.index(max(community_volumes))
        group_name = community_keywords[max_volume_idx]
        
        # Calculate total volume
        total_volume = sum(community_volumes)
        
        # Find the URL with most traffic for keywords in this group
        # Also track URLs even if they have 0 traffic
        url_traffic = {}
        url_appearances = {}
        
        for node in community_nodes:
            row = df.iloc[node]
            for i, url_col in enumerate(url_cols):
                url = row[url_col]
                traffic = row[traffic_cols[i]]
                
                if pd.notna(url) and url:
                    # Track traffic
                    if url not in url_traffic:
                        url_traffic[url] = 0
                        url_appearances[url] = 0
                    
                    if pd.notna(traffic):
                        url_traffic[url] += traffic
                    
                    # Track appearances (even with 0 traffic)
                    url_appearances[url] += 1
        
        # Get top URL - prioritize by traffic, but show any URL if all have 0 traffic
        if url_traffic:
            # Sort by traffic first, then by appearances as tiebreaker
            sorted_urls = sorted(url_traffic.keys(), 
                               key=lambda x: (url_traffic[x], url_appearances[x]), 
                               reverse=True)
            top_url = sorted_urls[0]
            total_traffic = url_traffic[top_url]
        else:
            top_url = ""
            total_traffic = 0
        
        groups_data.append({
            'Name': group_name,
            'Keywords': len(community_keywords),
            'Total Volume': total_volume,
            'Top URL': top_url,
            'Total Traffic': total_traffic,
            '_keywords': community_keywords  # Store for later use
        })
    
    # Convert to DataFrame
    result = pd.DataFrame(groups_data)
    
    # Sort by total volume descending
    if not result.empty:
        result = result.sort_values('Total Volume', ascending=False)
    
    return result

def process_keyword_group_details(df, selected_group, groups_df):
    """Process keyword details for a specific keyword group."""
    # Find the keywords in this group
    group_row = groups_df[groups_df['Name'] == selected_group]
    if group_row.empty:
        return pd.DataFrame(columns=['Keyword', 'Volume', 'KD', 'Competitor # Pages', 'Top Page', 'Traffic', 'Position'])
    
    group_keywords = group_row.iloc[0]['_keywords']
    
    # Filter the original data for these keywords
    filtered_df = df[df['Keyword'].isin(group_keywords)].copy()
    
    # Get competitor columns (excluding our site - first one)
    traffic_cols = [col for col in df.columns if 'Traffic' in col]
    position_cols = [col for col in df.columns if 'Position' in col and 'Average' not in col]
    url_cols = [col for col in df.columns if 'URL' in col]
    
    # Skip the first domain (our site) for competitor analysis
    if len(traffic_cols) > 1:
        competitor_traffic_cols = traffic_cols[1:]
        competitor_position_cols = position_cols[1:]
        competitor_url_cols = url_cols[1:]
    else:
        competitor_traffic_cols = traffic_cols
        competitor_position_cols = position_cols
        competitor_url_cols = url_cols
    
    result_rows = []
    
    for idx, row in filtered_df.iterrows():
        # Count competitor pages and find top page for this keyword
        competitor_count = 0
        max_traffic = 0
        top_page = ""
        top_position = 0
        
        for i, traffic_col in enumerate(competitor_traffic_cols):
            traffic_val = row[traffic_col]
            if pd.notna(traffic_val) and traffic_val > 0:
                competitor_count += 1
                if traffic_val > max_traffic:
                    max_traffic = traffic_val
                    top_page = row[competitor_url_cols[i]] if pd.notna(row[competitor_url_cols[i]]) else ""
                    top_position = row[competitor_position_cols[i]] if pd.notna(row[competitor_position_cols[i]]) else 0
        
        result_rows.append({
            'Keyword': row['Keyword'],
            'Volume': row['Volume'],
            'KD': row['KD'],
            'Competitor # Pages': competitor_count,
            'Top Page': top_page,
            'Traffic': max_traffic,
            'Position': top_position
        })
    
    result = pd.DataFrame(result_rows)
    
    # Sort by volume descending
    if not result.empty:
        result = result.sort_values('Volume', ascending=False)
    
    return result

def get_keyword_groups_sorted(groups_df):
    """Get keyword groups sorted by total volume."""
    if not groups_df.empty:
        return groups_df['Name'].tolist()
    return []

def get_competitor_pages_sorted(df):
    """Get competitor pages sorted by total traffic."""
    # Process to get all competitor pages
    competitor_pages_df = process_competitor_pages(df)
    if not competitor_pages_df.empty:
        return competitor_pages_df['URL'].tolist()
    return []

def get_queries_sorted_by_impressions(df):
    """Get unique queries sorted by total impressions (descending)."""
    query_impressions = df.groupby('Query')['Impressions'].sum().sort_values(ascending=False)
    return query_impressions.index.tolist()

def get_pages_sorted_by_impressions(df):
    """Get unique landing pages sorted by total impressions (descending)."""
    page_impressions = df.groupby('Landing Page')['Impressions'].sum().sort_values(ascending=False)
    return page_impressions.index.tolist()

def display_table_with_pagination(df, key_prefix, title):
    """Display a dataframe with export options and scrollable view."""
    st.subheader(title)
    
    # Export options row
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        # Download CSV button
        csv = df.to_csv(index=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{title.replace(' ', '_').replace(':', '')}_{timestamp}.csv"
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=filename,
            mime="text/csv",
            key=f"{key_prefix}_download"
        )
    
    with col2:
        # Copy to clipboard using a text area approach
        tsv = df.to_csv(sep='\t', index=False)
        st.text_area(
            "📋 Copy Table Data",
            value=tsv,
            height=100,
            key=f"{key_prefix}_copy_text",
            help="Click in the text area, press Ctrl+A (or Cmd+A on Mac) to select all, then Ctrl+C (or Cmd+C) to copy"
        )
    
    # Display row count info
    st.markdown(f"**Total rows: {len(df):,}**")
    
    # Prepare display dataframe
    df_display = df.copy()
    
    # Round numerical values for display but keep them as numbers
    if 'Average Position' in df_display.columns:
        df_display['Average Position'] = df_display['Average Position'].round(2)
    
    # Convert CTR to percentage if needed
    if 'CTR' in df_display.columns and df_display['CTR'].max() <= 1:
        df_display['CTR'] = df_display['CTR'] * 100
    
    # Configure column formatting for percentage columns while keeping numeric values
    column_config = {}
    
    if 'Cannibalisation %' in df_display.columns:
        column_config['Cannibalisation %'] = st.column_config.NumberColumn(
            'Cannibalisation %',
            format="%.2f%%",
            help="Percentage of impressions lost to cannibalization"
        )
    
    if 'Impression %' in df_display.columns:
        column_config['Impression %'] = st.column_config.NumberColumn(
            'Impression %',
            format="%.2f%%",
            help="Percentage of total query impressions"
        )
    
    if 'CTR' in df_display.columns:
        column_config['CTR'] = st.column_config.NumberColumn(
            'CTR',
            format="%.2f%%",
            help="Click-through rate"
        )
    
    # Calculate optimal height based on number of rows
    # Approximately 35px per row + 65px for header and padding
    num_rows = len(df_display)
    if num_rows == 0:
        display_height = 100  # Minimum height for empty table
    elif num_rows <= 20:
        # If 20 or fewer rows, size to fit exactly
        display_height = 35 * num_rows + 65
    else:
        # If more than 20 rows, fix at 20 rows height with scrolling
        display_height = 35 * 20 + 65  # 765px
    
    st.dataframe(
        df_display,
        use_container_width=True, 
        hide_index=True,
        height=display_height,
        column_config=column_config
    )

# Main Application
def main():
    # File Upload Section
    st.header("File Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Upload Cannibalization Export CSV**")
        cannibalization_file = st.file_uploader(
            "Choose Cannibalization Export file",
            type=['csv'],
            key="cannibalization_upload",
            help="Upload the CSV file containing Query, Landing Page, Impressions, Clicks, CTR, and Position data"
        )
        
        if cannibalization_file is not None:
            try:
                df = pd.read_csv(cannibalization_file)
                valid, result = validate_cannibalization_csv(df)
                
                if valid:
                    st.session_state['cannibalization_data'] = result
                    st.success(f"File loaded successfully! ({len(result)} rows)")
                else:
                    st.error(f"Validation error: {result}")
                    st.session_state['cannibalization_data'] = None
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.session_state['cannibalization_data'] = None
    
    with col2:
        st.markdown("**Upload Ahrefs Competitor Data CSV**")
        ahrefs_file = st.file_uploader(
            "Choose Ahrefs Competitor Data file",
            type=['csv'],
            key="ahrefs_upload",
            help="Upload the CSV file containing Keyword, Volume, KD, CPC, and competitor data"
        )
        
        if ahrefs_file is not None:
            try:
                df = pd.read_csv(ahrefs_file)
                valid, result = validate_ahrefs_csv(df)
                
                if valid:
                    st.session_state['ahrefs_data'] = result
                    st.success(f"File loaded successfully! ({len(result)} rows)")
                    
                    # Display detected competitor domains
                    competitor_cols = [col for col in result.columns if 'Traffic' in col]
                    domains = [col.split('/')[0] for col in competitor_cols]
                    if domains:
                        st.info(f"Detected domains: {', '.join(domains[:5])}{' and more' if len(domains) > 5 else ''}")
                else:
                    st.error(f"Validation error: {result}")
                    st.session_state['ahrefs_data'] = None
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.session_state['ahrefs_data'] = None
    
    # Process and display tables if cannibalization data is available
    if st.session_state['cannibalization_data'] is not None:
        st.markdown("---")
        
        # Process Current Keywords table
        current_keywords_df = process_current_keywords(st.session_state['cannibalization_data'])
        
        # Display metrics
        st.header("Analysis Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Keywords", f"{len(current_keywords_df):,}")
        with col2:
            st.metric("Total Clicks", f"{current_keywords_df['Clicks'].sum():,}")
        with col3:
            st.metric("Total Impressions", f"{current_keywords_df['Impressions'].sum():,}")
        with col4:
            avg_cannibalization = current_keywords_df['Cannibalisation %'].mean()
            st.metric("Avg Cannibalization", f"{avg_cannibalization:.1f}%")
        
        st.markdown("---")
        
        # Create tabs for Keywords, Pages, and optionally Competitor tabs and Keyword Groups
        if st.session_state['ahrefs_data'] is not None:
            if NETWORKX_AVAILABLE:
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Keywords", "Pages", "Competitor Keywords", "Competitor Pages", "Keyword Groups"])
            else:
                tab1, tab2, tab3, tab4 = st.tabs(["Keywords", "Pages", "Competitor Keywords", "Competitor Pages"])
        else:
            tab1, tab2 = st.tabs(["Keywords", "Pages"])
        
        with tab1:
            # Table 1: Current Keywords (in accordion, expanded by default)
            with st.expander("Table 1: Current Keywords", expanded=True):
                display_table_with_pagination(current_keywords_df, "current_keywords", "Current Keywords")
            
            st.markdown("---")
            
            # Table 2: Keyword Details (in accordion, minimized by default)
            with st.expander("Table 2: Keyword Details", expanded=False):
                st.subheader("Keyword Details")
                
                # Get queries sorted by impressions
                sorted_queries = get_queries_sorted_by_impressions(st.session_state['cannibalization_data'])
                
                selected_query = st.selectbox(
                    "Select a Query to view details:",
                    sorted_queries,
                    index=0,
                    key="query_selector",
                    help="Start typing to search for a specific query"
                )
                
                if selected_query:
                    keyword_details_df = process_keyword_details(
                        st.session_state['cannibalization_data'], 
                        selected_query
                    )
                    
                    # Display query metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Landing Pages", len(keyword_details_df))
                    with col2:
                        st.metric("Total Clicks", f"{keyword_details_df['Clicks'].sum():,}")
                    with col3:
                        st.metric("Total Impressions", f"{keyword_details_df['Impressions'].sum():,}")
                    
                    # Display the details table
                    display_table_with_pagination(keyword_details_df, "keyword_details", f"Details for: {selected_query}")
        
        with tab2:
            # Process Current Pages table
            current_pages_df = process_current_pages(st.session_state['cannibalization_data'])
            
            # Table 1: Current Pages (in accordion, expanded by default)
            with st.expander("Table 1: Current Pages", expanded=True):
                display_table_with_pagination(current_pages_df, "current_pages", "Current Pages")
            
            st.markdown("---")
            
            # Table 2: Page Details (in accordion, minimized by default)
            with st.expander("Table 2: Page Details", expanded=False):
                st.subheader("Page Details")
                
                # Get landing pages sorted by impressions
                sorted_pages = get_pages_sorted_by_impressions(st.session_state['cannibalization_data'])
                
                selected_page = st.selectbox(
                    "Select a Landing Page to view details:",
                    sorted_pages,
                    index=0,
                    key="page_selector",
                    help="Start typing to search for a specific landing page"
                )
                
                if selected_page:
                    page_details_df = process_page_details(
                        st.session_state['cannibalization_data'], 
                        selected_page
                    )
                    
                    # Display page metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Queries", len(page_details_df))
                    with col2:
                        st.metric("Total Clicks", f"{page_details_df['Clicks'].sum():,}")
                    with col3:
                        st.metric("Total Impressions", f"{page_details_df['Impressions'].sum():,}")
                    
                    # Display the details table
                    display_table_with_pagination(page_details_df, "page_details", f"Details for: {selected_page}")
        
        # Only show Competitor tabs if Ahrefs data is loaded
        if st.session_state['ahrefs_data'] is not None:
            with tab3:
                # Process Competitor Keywords table
                competitor_keywords_df = process_competitor_keywords(st.session_state['ahrefs_data'])
                
                # Table 1: Competitor Keywords (in accordion, expanded by default)
                with st.expander("Table 1: Competitor Keywords", expanded=True):
                    display_table_with_pagination(competitor_keywords_df, "competitor_keywords", "Competitor Keywords")
                
                st.markdown("---")
                
                # Table 2: Competitor Keyword Details (in accordion, minimized by default)
                with st.expander("Table 2: Competitor Keyword Details", expanded=False):
                    st.subheader("Competitor Keyword Details")
                    
                    # Get keywords sorted by volume where competitors rank
                    sorted_keywords = get_competitor_keywords_sorted(st.session_state['ahrefs_data'])
                    
                    if sorted_keywords:
                        selected_keyword = st.selectbox(
                            "Select a Keyword to view details:",
                            sorted_keywords,
                            index=0,
                            key="competitor_keyword_selector",
                            help="Start typing to search for a specific keyword"
                        )
                        
                        if selected_keyword:
                            competitor_details_df = process_competitor_keyword_details(
                                st.session_state['ahrefs_data'], 
                                selected_keyword
                            )
                            
                            # Get the keyword's volume and KD for metrics
                            keyword_info = st.session_state['ahrefs_data'][
                                st.session_state['ahrefs_data']['Keyword'] == selected_keyword
                            ].iloc[0] if not st.session_state['ahrefs_data'][
                                st.session_state['ahrefs_data']['Keyword'] == selected_keyword
                            ].empty else None
                            
                            # Display keyword metrics
                            if keyword_info is not None:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Volume", f"{int(keyword_info['Volume']):,}")
                                with col2:
                                    st.metric("KD", f"{keyword_info['KD']}")
                                with col3:
                                    st.metric("Competitor Pages", len(competitor_details_df))
                            
                            # Display the details table
                            display_table_with_pagination(competitor_details_df, "competitor_details", f"Details for: {selected_keyword}")
                    else:
                        st.info("No competitor data available. Please ensure the Ahrefs CSV contains competitor domains.")
            
            with tab4:
                # Process Competitor Pages table
                competitor_pages_df = process_competitor_pages(st.session_state['ahrefs_data'])
                
                # Table 1: Competitor Pages (in accordion, expanded by default)
                with st.expander("Table 1: Competitor Pages", expanded=True):
                    display_table_with_pagination(competitor_pages_df, "competitor_pages", "Competitor Pages")
                
                st.markdown("---")
                
                # Table 2: Competitor Page Details (in accordion, minimized by default)
                with st.expander("Table 2: Competitor Page Details", expanded=False):
                    st.subheader("Competitor Page Details")
                    
                    # Get competitor pages sorted by traffic
                    sorted_pages = get_competitor_pages_sorted(st.session_state['ahrefs_data'])
                    
                    if sorted_pages:
                        selected_page = st.selectbox(
                            "Select a URL to view details:",
                            sorted_pages,
                            index=0,
                            key="competitor_page_selector",
                            help="Start typing to search for a specific URL"
                        )
                        
                        if selected_page:
                            page_details_df = process_competitor_page_details(
                                st.session_state['ahrefs_data'], 
                                selected_page
                            )
                            
                            # Display page metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Keywords", len(page_details_df))
                            with col2:
                                st.metric("Total Traffic", f"{page_details_df['Traffic'].sum():,.0f}")
                            with col3:
                                st.metric("Total Volume", f"{page_details_df['Volume'].sum():,}")
                            
                            # Display the details table
                            display_table_with_pagination(page_details_df, "competitor_page_details", f"Details for: {selected_page}")
                    else:
                        st.info("No competitor page data available. Please ensure the Ahrefs CSV contains competitor domains.")
            
            if NETWORKX_AVAILABLE and st.session_state['ahrefs_data'] is not None:
                with tab5:
                    # Process keyword groups using cached function
                    # This will only compute once per unique dataset
                    df_hash = pd.util.hash_pandas_object(st.session_state['ahrefs_data']).sum()
                    
                    # Show a message if analysis is in progress
                    with st.spinner("Analyzing keyword relationships and creating groups..."):
                        keyword_groups_df = cached_process_keyword_groups(df_hash, st.session_state['ahrefs_data'])
                    
                    if keyword_groups_df is not None:
                        # Remove the internal _keywords column before display
                        display_groups_df = keyword_groups_df.drop(columns=['_keywords']) if '_keywords' in keyword_groups_df.columns else keyword_groups_df
                        
                        # Table 1: Keyword Groups (in accordion, expanded by default)
                        with st.expander("Table 1: Keyword Groups", expanded=True):
                            display_table_with_pagination(display_groups_df, "keyword_groups", "Keyword Groups")
                        
                        st.markdown("---")
                        
                        # Table 2: Keyword Group Details (in accordion, minimized by default)
                        with st.expander("Table 2: Keyword Group Details", expanded=False):
                            st.subheader("Keyword Group Details")
                            
                            # Get keyword groups sorted by volume
                            sorted_groups = get_keyword_groups_sorted(keyword_groups_df)
                            
                            if sorted_groups:
                                selected_group = st.selectbox(
                                    "Select a Keyword Group to view details:",
                                    sorted_groups,
                                    index=0,
                                    key="keyword_group_selector",
                                    help="Start typing to search for a specific keyword group"
                                )
                                
                                if selected_group:
                                    group_details_df = process_keyword_group_details(
                                        st.session_state['ahrefs_data'],
                                        selected_group,
                                        keyword_groups_df
                                    )
                                    
                                    # Get group info for metrics
                                    group_info = keyword_groups_df[keyword_groups_df['Name'] == selected_group].iloc[0]
                                    
                                    # Display group metrics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Keywords in Group", f"{int(group_info['Keywords']):,}")
                                    with col2:
                                        st.metric("Total Volume", f"{int(group_info['Total Volume']):,}")
                                    with col3:
                                        st.metric("Total Traffic", f"{group_info['Total Traffic']:,.0f}")
                                    
                                    # Display the details table
                                    display_table_with_pagination(group_details_df, "keyword_group_details", f"Details for: {selected_group}")
                            else:
                                st.info("No keyword groups found. This may happen if keywords don't share common ranking URLs.")
                    else:
                        st.error("Keyword groups analysis has not been completed yet.")
    
    else:
        # Show instructions if no data is loaded
        st.info("Please upload the Cannibalization Export CSV file to begin analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            SEO Keyword Research Tool | Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()