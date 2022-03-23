import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st
# from sympy import rot_axis3, rotations
# from torch import cosine_similarity
import gensim
# from gensim import corpora, models, similarities

# 1. Read data
# data = pd.read_csv('OnlineRetail.csv', encoding='unicode_escape')

#--------------
# GUI
st.title("Data Science Project")
st.write("## Recommendation System")

# Upload file/ Read file
products = pd.read_csv("data/Product_new.csv")
reviews = pd.read_csv("data/Review_new.csv")
# select box for product name
product_name = products[['item_id','name']]
CHOICES = dict(product_name.values)
def format_func(option):
        return CHOICES[option]
def display_image(select_product):
        col = st.columns(2)
        with col[0]:
            st.image(select_product['image'][0])
        with col[1]:
            formatted_string = "{:,.0f} VNĐ".format(select_product['price'][0])
            st.write(select_product['name'][0])
            st.write("Thương hiệu: ",select_product['brand'][0])
            st.write("Giá: ",formatted_string)         
# 2. Data pre-processing


# 3. Build model


#5. Save models
# luu model classication
# pkl_filename = "ham_spam_model.pkl"  
# with open(pkl_filename, 'wb') as file:  
#     pickle.dump(model, file)
  
# luu model CountVectorizer (count)
# pkl_count = "count_model.pkl"  
# with open(pkl_count, 'wb') as file:  
#     pickle.dump(count, file)


#6. Load models 
# Đọc model
# import pickle
# with open(pkl_filename, 'rb') as file:  
#     ham_spam_model = pickle.load(file)
# # doc model count len
# with open(pkl_count, 'rb') as file:  
#     count_model = pickle.load(file)

# GUI
menu = ["Business Objective", "EDA", "Content-based / Cosine Similarity", "Content-based / Gensim", "Collaborative filtering"]

choice = st.sidebar.radio('Select option', menu)
if choice == "Business Objective" :
    st.subheader("Business Objective")
     
    st.write("""###### => Problem: Xây dựng Recommendation System cho một hoặc một số nhóm hàng hóa trên tiki.vn giúp đề xuất và gợi ý cho người dùng/ khách hàng.
###### => Requirement: Xây dựng các mô hình đề xuất:
- Content-based filtering
- Collaborative filtering
""")
    st.image("images/reco_01.png")
elif choice == "EDA" :
    st.subheader("EDA (Exploratory Data Analysis)")
    # 1. Products
    st.write("## 1. Products")
    ## Some data
    st.write("#### 1.1 Some data")
    st.dataframe(products.head(3))
    st.dataframe(products.tail(3))
    ## Rating, price describe
    st.write("#### 1.2 Rating, Price describe")
    pd.options.display.float_format = '{:,.2f}'.format
    st.dataframe(products[['rating','price']].describe())
    st.write(
        """
        #### Nhận xét:
        - Rating trong khoảng 0-5
        - Giá sản phẩm dao động rất rộng từ 7.000 cho đến 62.690.000
        """
    )
    ## Phân tích giá bán
    p = np.percentile(products.price, 75)
    formatted_string = "{:,.2f}".format(p)
    st.write("#### 1.3 Phân tích giá bán")
    st.image("images/phan_tich_gia_ban.png")
    st.write(
        """
        #### Nhận xét:
        - Giá phân phối lệch phải
        - Dao động lớn 7.000 - 62.690.000
        """
    )
    st.write("- Đa số giá sản phẩm < ", formatted_string)
    ## Phân tích thương hiệu
    brands = products.groupby('brand')['item_id'].count().sort_values(ascending=False)
    st.write("#### 1.4 Thương hiệu")
    st.write(brands)
    st.write(
        """
        #### Nhận xét:
        - Thương hiệu có số lượng sản phẩm cao nhất là OEM, đây không phải là thương hiệu công ty mà chỉ ra các sản phẩm được sản xuất bởi nhà cung cấp gốc. Do đó OEM không đưa vào phân tích thương hiệu
        """
    )
    # Top 10 thương hiệu có số lượng mã sản phẫm cao nhất
    # brands[1:11].plot(kind='bar')
    # plt.ylabel('Count')
    # plt.title('Product items by brand - Top 10')
    # plt.xticks(rotation = 15)
    # plt.savefig('images/top10_brands.png')
    st.image('images/top10_brands.png')
    st.write(
        """
        #### Nhận xét:
        - Samsung có nhiều sản phẩm nhất (~200), các thương hiệu khác thì tương đương nhau khoảng 75 - 100 sản phẩm
        """
    )
    # Giá bán theo thương hiệu
    # price_by_brand = products.groupby(by='brand').mean()['price']
    # price_by_brand.sort_values(ascending=False)[:10].plot(kind='bar')
    # plt.ylabel('Price')
    # plt.title('Avg price by brand')
    # plt.xticks(rotation = 15)
    # plt.savefig('images/avg_price_by_brand.png')
    st.image('images/avg_price_by_brand.png')
    st.write(
        """
        #### Nhận xét:
        - Hitachi có giá trung bình cao nhất
        """
    )
    ## Rating
    # sns.displot(products, x='rating', kind='hist')
    # plt.title('Product rating histogram')
    # plt.savefig("images/rating.png")
    st.write("#### 1.5 Rating")
    st.image('images/rating.png')
    st.write(
        """
        #### Nhận xét:
        - Đa số sản phẩm có rating là 0 và 5, 2 rating này tương đương nhau
        - Đa số sản phẩm có rating > 4
        """
    )
    # Product - Rating
    # avg_rating_customer = reviews.groupby(by='product_id').mean()['rating'].to_frame().reset_index()
    # avg_rating_customer.rename({'rating':'avg_rating'}, axis=1, inplace=True)
    # products = products.merge(avg_rating_customer, left_on='item_id', right_on='product_id', how='left')
    # fig, ax = plt.subplots(1,2, figsize=(15,10))
    # sns.displot(products, x='rating', kind='hist', ax=ax[0])
    # sns.displot(products, x='avg_rating', kind='hist', ax=ax[1])
    # plt.title('Review rating histogram')
    # plt.savefig('images/rating_vs_avgrating.png')
    st.image('images/rating_vs_avgrating.png')
    st.write(
        """
        #### Nhận xét:
        - Rating sản phẩm trong review của khách hàng bắt đầu từ 1. Có thể kết luận điểm rating = 0 trong product là do thiếu dữ liệu
        """
    )
    # 2. Reviews
    st.write("## 2. Reviews")
    st.dataframe(reviews.head())
    # st.dataframe(data[['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice', 'CustomerID']].head(3))
    # st.dataframe(data[['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice', 'CustomerID']].tail(3))
    
    st.write("#### 2.1 Overview")
    product_reviews = reviews.groupby('product_id').count().shape[0]
    st.write('Có ', reviews.shape[0], ' đánh giá cho ', product_reviews, ' sản phẩm')

    st.write("#### 2.2 Rating")
    # sns.displot(reviews, x='rating', kind='kde')
    # plt.savefig("images/review_rating.png")
    st.image("images/review_rating.png")
    st.write(
        """
        #### Nhận xét:
        - Đa số phản hồi đánh giá 4 hoặc 5 --> tích cực
        """
    )
   
    st.write("#### 2.3 Top 20 sản phẩm được đánh giá nhiều nhất")
    # plt.figure(figsize=(12,8))
    # top_products = reviews.groupby('product_id').count()['customer_id'].sort_values(ascending=False)[:20]
    # top_products.index = products[products.item_id.isin(top_products.index)]['name'].str[:25]
    # top_products.plot(kind='bar')
    # plt.xticks(rotation = 15)
    # plt.savefig("images/review_top_20_product.png")
    st.image("images/review_top_20_product.png")
    st.write(
        """
        #### Nhận xét:
        - Chuột không dây Logitech được đánh giá nhiều nhất, nhiều gấp đôi so với sản phẩm tiếp theo.
        """
    )

    st.write("#### 2.4 Top 20 khách hàng đánh giá nhiều nhất")
    # top_rating_customers = reviews.groupby('customer_id').count()['product_id'].sort_values(ascending=False)[:20]
    # plt.figure(figsize=(12,6))
    # plt.bar(x=[str(x) for x in top_rating_customers.index], height=top_rating_customers.values)
    # plt.xticks(rotation=15)
    # plt.savefig("images/review_top_20_customer.png")
    st.image("images/review_top_20_customer.png")
    st.write(
        """
        #### Nhận xét:
        - Khách hàng 7737978 tích cực đánh giá sản phẩm nhất.
        """
    )  
    
elif choice == 'Content-based / Cosine Similarity':
              
    st.subheader("Content based - Cosine Similarity")
    st.image("images/cosin.png")
    st.write("#### Cosine Similarity result")
    cosine_similarity_data = pd.read_csv('data/CB_new.csv')
    cosine_similarity_data = cosine_similarity_data.iloc[: , 1:]
    st.dataframe(cosine_similarity_data.head())
    st.write("#### Predict base on product id:")
    # Select box
   
    
    option = st.selectbox("Select product", options=list(CHOICES.keys()), format_func=format_func)
    select_product = products[products['item_id'] == option].reset_index()
    # Hiển thị select product
    display_image(select_product)    
    # Results
    similar_products = cosine_similarity_data[cosine_similarity_data['product_id'] == option]
    similar_products['rcmd_product_name'] = similar_products.rcmd_product_id.map(CHOICES)
    similar_products = pd.merge(similar_products,products,left_on=['rcmd_product_id'], right_on = ['item_id'], how = 'left')


    st.write("### Similarity products:")
    # st.dataframe(similar_products)
    for index, row in similar_products.iterrows():
        row = row.to_frame().T.reset_index()
        display_image(row)

elif choice == 'Content-based / Gensim':
    # Reset index for product
    products = products.reset_index()
    # Load model
    dictionary = gensim.corpora.dictionary.Dictionary.load("data/dictionary_model")
    tfidf = gensim.models.tfidfmodel.TfidfModel.load("data/tfidf_model")
    index = gensim.similarities.docsim.SparseMatrixSimilarity.load("data/index_model")
    # Gensim recommender
    def recommender(view_product, dictionary, tfidf, index):
        # Convert search words into Sparse Vectors
        view_product = view_product.lower().split()
        kw_vector = dictionary.doc2bow(view_product)
        # similarity calculation
        sim = index[tfidf[kw_vector]]

        # Print result
        list_id = []
        list_score = []
        for i in range(len(sim)):
            list_id.append(i)
            list_score.append(sim[i])

        df_result = pd.DataFrame({'id': list_id,
                                'score': list_score})

        # five highest scores
        five_highest_score = df_result.sort_values(by='score', ascending=False).head(6)
        idToList = list(five_highest_score['id'])

        products_find = products[products.index.isin(idToList)]
        results = products_find[['index', 'item_id', 'name']]
        results = pd.concat([results, five_highest_score], axis=1).sort_values(by='score', ascending=False)
        return results
    # Display
    st.subheader("Content based - Gensim")
    st.image("images/gensim.png")
    st.write("#### Predict base on Gensim")
    option = st.selectbox("Select product", options=list(CHOICES.keys()), format_func=format_func)
    select_product = products[products['item_id'] == option].reset_index()
    # Hiển thị select product
    display_image(select_product)
    text = select_product['name_description_pre'].to_string(index=False)
    # Gensim predict
    results = recommender(text, dictionary, tfidf, index)
    results = results[['item_id', 'name', 'score']]
    similar_products = pd.merge(results,products,left_on=['item_id'], right_on = ['item_id'], how = 'left')
    similar_products.rename(columns = {'name_x':'name'}, inplace = True)    
    # # Results
    st.write("#### Similarity products:")
    # st.dataframe(similar_products)
    # Display result
    for index, row in similar_products.iterrows():
        row = row.to_frame().T.reset_index()
        display_image(row)

elif choice == 'Collaborative filtering':
    # Display
    st.subheader("Collaborative filtering")
    # st.image("images/collaborative.png")
    # ALS Model
    st.write('''
             #### ALS - Alternating Least Squares matrix factorization
             - Chúng ta sẽ đưa ra các đề nghị sản phẩm cho tất cả User bằng model ALS.
             ''')
    st.image("images/ALS_1.png")
    st.image("images/ALS_2.png")
    st.write('''RMSE - Root Mean Square Error có 1 kết quả không mấy khả quan tại lần đầu train cho bộ dữ liệu.''')
    st.image("images/ALS_3.png")
    st.write('''RMSE sau nhiều thử nghiệm đã có 1 kết quả tốt hơn. Chúng ta sẽ sử dụng kết quá này để dự báo cho toàn bộ user trong bộ dữ liệu.''')
    user_recs = pd.read_csv('data/user_recs_100.csv')
    # Select box 
    customer_id = user_recs['customer_id'].drop_duplicates()
    customer_id_choice = st.selectbox('Please choose a user to see the recommended items::', customer_id)
    recs = user_recs[user_recs['customer_id'] == customer_id_choice]
    recs = recs.merge(products, on=['item_id'], how='left')

    st.markdown("<h4 style='text-align: left; color: #339966; '>Các sản phẩm đề nghị cho người dùng này</h4>", unsafe_allow_html=True)
    # st.dataframe(recs)
    def display_image(select_product):
        col = st.columns(2)
        with col[0]:
            st.image(select_product['image'][0])
        with col[1]:
            formatted_string = "{:,.0f} VNĐ".format(select_product['price'][0])
            st.write(select_product['name'][0])
            st.write("Thương hiệu: ",select_product['brand'][0])
            # st.write("Miêu tả sản phẩm: ",select_product['description_y'][0])
            st.write("Giá: ",formatted_string)
    for index, row in recs.iterrows():
        row = row.to_frame().T.reset_index()
        display_image(row)

