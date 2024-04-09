import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import streamlit as st

# Define non-binary labels globally
non_binary_labels = [
    'Messiness', 'Noise Level', 'Temperature (F)',
    'Alcohol Consumption', 'Expected Bed Time', 'Expected Wake Time'
]

def load_data(file_path):
    data = {}
    try:
        dataframe = pd.read_csv(file_path)
        for _, row in dataframe.iterrows():
            name = row['Name']
            for label in non_binary_labels:
                row[label] = float(row[label])  # Ensure all non-binary labels are float
            data[name] = row.to_dict()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    return data, dataframe

# New optimize_matching function
def optimize_matching(data, weights):
    m = gp.Model()
    x = m.addVars(data.keys(), data.keys(), vtype=GRB.BINARY, name='x')

    # Calculate ranges for each label to use in normalization
    ranges = {}
    for label in non_binary_labels:
        all_values = [data[name][label] for name in data.keys()]
        ranges[label] = max(all_values) - min(all_values) if max(all_values) != min(all_values) else 1

    # Objective function: Minimize normalized incompatibility scores
    m.setObjective(
        gp.quicksum(
            weights[label] * ((data[i][label] - data[j][label]) / ranges[label]) ** 2 * x[i, j]
            for i in data.keys()
            for j in data.keys() if i != j
            for label in non_binary_labels
        ),
        GRB.MINIMIZE
    )

    # Constraints based on preferences
    for i in data.keys():
        for j in data.keys():
            if i != j:
                # Add constraints only when preferences are not compatible
                if data[i]['Biological Sex'] != data[j]['Biological Sex']:
                    m.addConstr(x[i, j] == 0, name=f"sex_constraint_{i}_{j}")
                if data[i]['Overnight Guests'] != data[j]['Overnight Guests']:
                    m.addConstr(x[i, j] == 0, name=f"guests_constraint_{i}_{j}")

    # Ensure at least one match for each student
    for i in data.keys():
        m.addConstr(gp.quicksum(x[i, j] for j in data.keys() if i != j) >= 1)

    m.optimize()

    # Extract and sort matches based on compatibility score
    top_matches = {i: [] for i in data.keys()}
    for i in data.keys():
        for j in data.keys():
            if i != j and x[i, j].X > 0.5:  # If they are a potential match
                score = sum(
                    weights[label] * (data[i][label] - data[j][label]) ** 2
                    for label in non_binary_labels
                )
                top_matches[i].append((j, score))
        top_matches[i] = sorted(top_matches[i], key=lambda item: item[1])[:5]
    
     # After computing the compatibility scores
    max_score = max([max(match[1] for match in matches) for matches in top_matches.values()]) if top_matches else 1
    for i in top_matches.keys():
        for j in range(len(top_matches[i])):
            student, score = top_matches[i][j]
            # Inverting and normalizing the score
            compatibility_percentage = 100 * (1 - (score / max_score))
            top_matches[i][j] = (student, round(compatibility_percentage, 2))

    return top_matches

# Function to optimize matching with fixed pairs
def optimize_matching_with_fixed_pairs(data, weights, fixed_pairs):
    m = gp.Model()
    x = m.addVars(data.keys(), data.keys(), vtype=GRB.BINARY, name='x')

    # Normalization ranges
    ranges = {label: max(all_values) - min(all_values) if max(all_values) != min(all_values) else 1 
              for label in non_binary_labels 
              for all_values in ([data[name][label] for name in data.keys()],)}

    # Objective function: Minimize normalized incompatibility scores
    m.setObjective(
        gp.quicksum(
            weights[label] * ((data[i][label] - data[j][label]) / ranges[label]) ** 2 * x[i, j]
            for i in data.keys()
            for j in data.keys() if i != j
            for label in non_binary_labels
        ),
        GRB.MINIMIZE
    )

    # Constraints based on preferences and fixed pairs
    fixed_individuals = {person for pair in fixed_pairs for person in pair}
    for i in data.keys():
        for j in data.keys():
            if i != j:
                # Skip adding constraints for fixed pairs
                if (i, j) in fixed_pairs or (j, i) in fixed_pairs:
                    continue
                
                # Add constraints only when preferences are not compatible
                if data[i]['Biological Sex'] != data[j]['Biological Sex']:
                    m.addConstr(x[i, j] == 0, name=f"sex_constraint_{i}_{j}")
                if data[i]['Overnight Guests'] != data[j]['Overnight Guests']:
                    m.addConstr(x[i, j] == 0, name=f"guests_constraint_{i}_{j}")

    # Ensure at least one match for each student not in fixed pairs
    for i in data.keys():
        if i not in fixed_individuals:
            m.addConstr(gp.quicksum(x[i, j] for j in data.keys() if i != j) >= 1)

    m.optimize()

    # Extract results, excluding fixed individuals from the optimization results
    top_matches = {i: [] for i in data.keys() if not any(i in pair for pair in fixed_pairs)}
    for i in top_matches.keys():
        for j in data.keys():
            if j not in fixed_individuals and i != j and x[i, j].X > 0.5:
                score = sum(weights[label] * ((data[i][label] - data[j][label]) / ranges[label])**2 
                            for label in non_binary_labels)
                top_matches[i].append((j, score))
        top_matches[i] = sorted(top_matches[i], key=lambda item: item[1])[:5]

    # Normalize the scores for other matches
    max_score = 1
    if any(matches for matches in top_matches.values()):
        max_score = max([max(match[1] for match in matches) for matches in top_matches.values() if matches])

    for i in top_matches.keys():
        for j in range(len(top_matches[i])):
            student, score = top_matches[i][j]
            compatibility_percentage = 100 * (1 - (score / max_score))
            top_matches[i][j] = (student, round(compatibility_percentage, 2))

    return fixed_pairs, top_matches

# Initialize session state variables
if 'show_matches' not in st.session_state:
    st.session_state['show_matches'] = False
if 'show_fixed_pairs' not in st.session_state:
    st.session_state['show_fixed_pairs'] = False
if 'optimize_results' not in st.session_state:
    st.session_state['optimize_results'] = None
if 'fixed_pairs_count' not in st.session_state:
    st.session_state['fixed_pairs_count'] = 0
if 'fixed_pairs' not in st.session_state:
    st.session_state['fixed_pairs'] = []

def get_color_for_score(score):
    # This function returns a color from red to green based on the score
    green = int(score * 2.55)
    red = 255 - green
    return f"rgb({red},{green},0)"

def display_results(other_matches, fixed_pairs=None):
    # Display Fixed Pairs
    if fixed_pairs:
        st.markdown("#### Fixed Pairs")
        fixed_pairs_df = pd.DataFrame(fixed_pairs, columns=['Student 1', 'Student 2'])
        st.write(fixed_pairs_df.to_html(index=False), unsafe_allow_html=True)

    # Display Other Matches with color-coded compatibility scores
    if other_matches:
        st.markdown("#### Other Matches")
        num_cols = 2  # Adjust as needed
        # Sort the other_matches dictionary by student names
        sorted_matches = dict(sorted(other_matches.items(), key=lambda x: x[0]))

        match_iterator = iter(sorted_matches.items())

        while True:
            cols = st.columns(num_cols)
            try:
                for i in range(num_cols):
                    student, matches = next(match_iterator)
                    html_content = f"<b>{student}'s Top Matches:</b><br><table>"
                    html_content += "<tr><th>Match</th><th>Compatibility (%)</th></tr>"
                    for match, score in matches:
                        color = get_color_for_score(score)  # Function to determine the color
                        html_content += f"<tr><td>{match}</td><td style='color: {color}'>{score}%</td></tr>"
                    html_content += "</table>"
                    with cols[i]:
                        st.markdown(html_content, unsafe_allow_html=True)
            except StopIteration:
                break

# Streamlit UI code
st.header("Upload Data")
st.write("Upload your CSV file with student preferences for roommate matching.")
uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file is not None:
    data, df = load_data(uploaded_file)
    if data and isinstance(data, dict) and len(data) > 0:
        #Displaying the headers (column names)
        st.header('2. Selected Mandatory Categories')
        selected_headers = st.multiselect("From the categories below, select those that are mandatory for roommate selection (i.e. Sex/Gender)", options=[col for col in df.columns if col != 'Name'])
        if selected_headers:
            non_binary_labels = [header for header in df.columns.tolist() if header not in selected_headers and header != 'Name']
        st.header("Enter Parameters")
        st.write("Set the weights for each preference to customize the matching algorithm. Please select values between 0 and 1 to indicate the importance of each parameter.")
        weights = {}
        for label in non_binary_labels:
            weight = st.number_input(f"Weight for '{label}':", min_value=0.0, max_value=1.0, step=0.1, format="%.1f", key=label)
            if weight < 0.0 or weight > 1.0:
                st.error("Invalid weight: Please enter a value between 0.0 and 1.0.")
            weights[label] = weight

        st.header("Optimize Roommate Matching")
        st.write("Click the button to find the best roommate matches based on the preferences and weights.")

        if st.button("Optimize Matching"):
            optimized_matches = optimize_matching(data, weights)
            st.session_state['optimize_results'] = optimized_matches
            st.session_state['show_matches'] = True

        # Display optimization results
        if st.session_state['show_matches'] and st.session_state['optimize_results']:
            display_results(st.session_state['optimize_results'])

        # Section for adding fixed pairs
        if st.button("Add Fixed Pair"):
            if st.session_state['fixed_pairs_count'] < 5:
                st.session_state['fixed_pairs_count'] += 1
            else:
                st.warning("Maximum of 5 fixed pairs allowed.")

        if st.button("Remove Last Fixed Pair"):
            if st.session_state['fixed_pairs_count'] > 0:
                st.session_state['fixed_pairs_count'] -= 1
                if st.session_state['fixed_pairs']:
                    st.session_state['fixed_pairs'].pop()

        students = sorted(list(data.keys())) if data else []
        # Collecting selected pairs in a temporary list
        temp_pairs = []
        for i in range(st.session_state['fixed_pairs_count']):
            col1, col2 = st.columns(2)
            with col1:
                student1 = st.selectbox(f"Pair {i+1} - Student 1:", [''] + students, key=f'student1_{i}')
            with col2:
                student2 = st.selectbox(f"Pair {i+1} - Student 2:", [''] + students, key=f'student2_{i}')

            if student1 and student2:
                temp_pairs.append((student1, student2))

        # Validate and add new pairs
        for student1, student2 in temp_pairs:
            if student1 == student2:
                st.error("Invalid pair: Same student selected for both slots. Please select different students.")
                break
            elif any((pair[0] == student1 and pair[1] == student2) or (pair[0] == student2 and pair[1] == student1) for pair in st.session_state['fixed_pairs']):
                st.error("Invalid pair: This pair has already been selected.")
                break
            else:
                # Add the pair if it's a new and valid combination
                if not any(student1 in pair or student2 in pair for pair in st.session_state['fixed_pairs']):
                    st.session_state['fixed_pairs'].append((student1, student2))

        if st.session_state['fixed_pairs'] and st.button("Optimize Matching with Fixed Pairs"):
            fixed_pairs, other_matches = optimize_matching_with_fixed_pairs(data, weights, st.session_state['fixed_pairs'])
            display_results(other_matches, fixed_pairs)

    else:
        st.error("No data loaded or data format is incorrect.")