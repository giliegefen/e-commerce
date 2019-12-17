בתחילה ניסינו להשתמש בmovie similarity עם חישוב cosine על r טילדה (r האמיתי - r שמצאנו בחלק 2)
ורצינו לקחת את k השכנים הקרובים וכמו שרואים בהרצאה לחשב את המכפלה ולהוסיף לפרדיקציה מ-2.
בגלל מגבלת זמן ושלקח למחשב המון זמן לרוץ לא הצלחנו לדבג עד הסוף את הפונקציה או להשתמש בה.
בסוף השתמשנו פשוט בפרדיקציה של סעיף 2 בחלק 1 כי היא נתנה לנו את השגיאה המינימלית.


הפעלת סעיף 2:

data = st.ModelData('allX.csv', 'allY.csv', 'comp_set.csv', 'test_y.G.csv', 'movie_data.csv')

r_avg = st.calc_average_rating(data.train_y)


c = st.construct_rating_vector(data.train_y, r_avg)
A1 = st.create_coefficient_matrix(data.train_x, data)
b1 = st.fit_parameters(A1, c)
test_predictions = st.model_inference(data.test_x, b1, r_avg, data)
data.test_x['r_hat'] = test_predictions
print(test_predictions)

ניסיון לדימיון משתמש-משתמש:

def calc(k):
    header = ['user', 'movie', 'rate']
    df = pd.read_csv('dataset.csv', sep=',', names=header)
    train_data, test_data = cv(df, test_size=0.20)


    n_users = df.user.unique().shape[0]
    n_items = df.movie.unique().shape[0]
    users=defaultdict(list)

    for x, row in train_data.iterrows():
        users[row[0]]+=[row[1],row[2]]

    train_data_matrix = np.zeros((4702, 45337))
    for line in train_data.itertuples():
        if(line[0]!=0):
            train_data_matrix[int(line[1])-1, int(line[2])-1] = line[3]

    u=[]
    uu={}
    t=1
    user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
    for line in np.array(user_similarity):
        uu[t]=line.argsort()[:k]
        t+=1
    print('hi')

    rate_u=defaultdict(list)
    for x, row in test_data.iterrows():
        if row[0]!='user':
            kk=(uu[int(row[0])])
            rate = 0.0
            c=0
            for i in range(1,k):
                y=(train_data_matrix[kk[i],int(row[1])-1])
                if y>0:
                    rate=rate+y
                    c+=1
            if c==0: c=1
            rate_u[row[0]] += [row[1],(rate / c)]

    print('hi')

    users_test=defaultdict(list)

    for x, row in test_data.iterrows():
        users_test[row[0]]+=[row[1],row[2]]
    mse=0
    tollmse=0
    r_mse={}
    for ij in users_test:
        if ij=='user':
            continue
        rmse=0
        cc=0
        r1 = (users_test[ij])
        r2 = (rate_u[ij])
        for ii in range(0,len(r1)-1):
            if r1[ii]==r2[ii]:
                rmse += ((float(r1[ii + 1]) - float(r2[ii + 1])) ** 2)
                cc+=1
                ii+=1
        if cc==0: cc=1
        r_mse[ij]=(math.sqrt(rmse/cc))
        tollmse+=r_mse[ij]
    mse=tollmse/len(users_test)
    return mse

ניסיון לדמיון סרט-סרט:

data = st.ModelData('train_x.G.csv', 'train_y.G.csv', 'test_x.G.csv', 'test_y.G.csv', 'movie_data.csv')

r_avg = st.calc_average_rating(data.train_y)


c = st.construct_rating_vector(data.train_y, r_avg)
A1 = st.create_coefficient_matrix(data.train_x, data)
b1 = st.fit_parameters(A1, c)
test_predictions = st.model_inference(data.train_x, b1, r_avg, data)
data.train_x['r_hat'] = test_predictions
data.train_x['r_tilda'] = data.train_y['rate']-data.train_x['r_hat']

train_data_matrix = np.zeros((10, 5))
train_data_matrix[data.train_x['user']-1, data.train_x['movie']-1] = data.train_x['r_tilda']


uu={}
t=1
k=2


item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')


for line in np.array(item_similarity):
    uu[t] = line.argsort()[:k]
    t += 1

rate_u=defaultdict(list)
for x, row in data.test_x.iterrows():
    kk=(uu[int(row[1])])
    rate = 0.0
    c=0
    for i in range(1,k):
        y=(train_data_matrix[kk[i],int(row[1])-1])
        if y>0:
            rate=rate+y
            c+=1
        if c==0: c=1
        rate_u[row[0]] += [row[1],(rate / c)]