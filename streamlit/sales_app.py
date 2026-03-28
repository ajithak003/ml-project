import pandas as pd
import streamlit as st

#Task 1 — Build the App

st.title("Sales Dashboard")
st.subheader("Welcome to the Sales Dashboard!")
st.write("This dashboard provides insights into sales data. You can explore various visualizations and metrics to understand sales performance.")

sales_data = pd.DataFrame({
    "Product": ["Honeycrisp Apples", "Organic Bananas", "Frozen Salmon", "Dark Chocolate", "Espresso Beans"],
    "Category": ["Fruit", "Fruit", "Seafood", "Snacks", "Beverage"],
    "Sales": [1240, 980, 760, 430, 590],
})

#Task 2 — Add a Sidebar
category_options = ["All Categories"] + sorted(sales_data["Category"].unique())
category_filter = st.sidebar.selectbox("Filter by category", category_options)

if category_filter == "All Categories":
    displayed_data = sales_data
else:
    displayed_data = sales_data[sales_data["Category"] == category_filter]

st.dataframe(displayed_data)
st.line_chart(displayed_data.set_index("Product")["Sales"])