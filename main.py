import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# Veri setini yükle
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
df = pd.read_csv(url, header=None)

# Veri setinin son birkaç satırını doğru yüklendiğinden emin olmak için göster
print("Veri setinin sonu:")
print(df.tail())

# Verilerin ön işlemesi
# Özellikleri ve etiketleri çıkarma
X = df.loc[:, 2:].values
y = df.loc[:, 1].values

# Kategorik etiketleri kodlama ('M' = malign, 'B' = benign)
le = LabelEncoder()
y = le.fit_transform(y)
print("\nKodlama sonrası sınıflar:", le.classes_)

# Veri setini eğitim (%80) ve test (%20) olarak bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

# Özellikleri standartlaştır
scaler = StandardScaler()

# PCA dönüşümü
pca = PCA(n_components=2)  # Görselleştirme amaçları için 2 bileşene indirge

# Lojistik Regresyon modeli
logistic = LogisticRegression(random_state=1, max_iter=10000)

# Standartlaştırma, PCA uygulama ve ardından lojistik regresyonu sığdırmak için bir boru hattı oluştur
pipeline = make_pipeline(scaler, pca, logistic)

# Eğitim verileri üzerinde boru hattını eğit
pipeline.fit(X_train, y_train)

# Modeli bir dosyaya kaydet
model_filename = 'trained_model.pkl'
joblib.dump(pipeline, model_filename)
print(f"\nEğitilmiş model '{model_filename}' dosyasına kaydedildi.")

# Test verileri üzerinde modeli değerlendir
accuracy = pipeline.score(X_test, y_test)
print("\nPCA ile Test Verileri Üzerinde Model Doğruluğu:", accuracy)

# İlk iki ana bileşeni çiz
plt.figure(figsize=(10, 6))
X_train_pca = pca.transform(scaler.transform(X_train))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='coolwarm', edgecolor='k', s=50)
plt.title('PCA ile Projeksiyon')
plt.xlabel('Ana Bileşen 1')
plt.ylabel('Ana Bileşen 2')
plt.colorbar()
plt.show()

# Modeli bir dosyadan yükle ve test doğruluğunu kontrol et
loaded_pipeline = joblib.load(model_filename)
loaded_accuracy = loaded_pipeline.score(X_test, y_test)
print("\nYüklenen model ile Test Verileri Üzerinde Model Doğruluğu:", loaded_accuracy)
