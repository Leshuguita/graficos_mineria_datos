import pandas as pd

print("Cargando transacciones...")
# archivo completo
tdf = pd.read_csv("./credit_card_transactions-ibm_v2.csv")
# solo user 0 (mas rapido, para probar cosas)
#tdf = pd.read_csv("./User0_credit_card_transactions.csv")
print("Cargando tarjetas...")
cdf = pd.read_csv("./sd254_cards.xls")

print("Trabajando...")
fraud_per_card = tdf[["User", "Card", "Is Fraud?"]].copy()
fraud_per_card["Is Fraud?"] = fraud_per_card["Is Fraud?"].map({'Yes':True, 'No':False})
fraud_per_card = fraud_per_card.groupby(["User","Card"]).sum()
fraud_per_card = fraud_per_card.rename(columns={"Is Fraud?":"Fraud Count"})

joined = cdf.rename(columns={"CARD INDEX":"Card"}).join(fraud_per_card, on=["User", "Card"])

# print(joined.corr(numeric_only=True))

# Las unica leves correlaciones parecen ser con la ultima vez que se cambio el pin, y la cantidad de plasticos. Podria ser interesante ver
# el cambio entre antes/despues del cambio de pin

# No hay fuertes diferencias para frautes/targeta entre cantidad de copias, ni entre tipos de tarjetas

print("Dibujando gráficos...")

# grafico de torta de promedio de fraudes
# segun tipo de tarjeta
mean = joined[["Card Type", "Fraud Count"]].groupby("Card Type").mean()
plot = mean.plot.pie(y="Fraud Count", legend=False, autopct="%1.1f%%",ylabel="", title="Promedio de fraudes por tipo de tarjeta")
fig = plot.get_figure()
fig.savefig("./by_card_type.svg")

# lo mismo que antes pero boxplots,
# no aporta mucho mas que el de torta la verdad
group = joined[["Card Type", "Fraud Count"]].groupby("Card Type")
by_type = pd.concat([pd.Series(v['Fraud Count'].tolist(), name=k) for k, v in group], axis=1)
plot = by_type.plot.box(vert=False)
fig = plot.get_figure()
fig.savefig("./by_card_type_box.svg")

# promedio de frauded segun cantidad de copias
# de la tarjeta
plot = joined[["Cards Issued", "Fraud Count"]].groupby("Cards Issued").mean().plot.bar(title="Promedio de fraudes por cantidad de copias de tarjeta")
fig = plot.get_figure()
fig.savefig("./by_num_copies.svg")

# no hay nada en la dark web xd
# plot = joined[["Card on Dark Web", "Fraud Count"]].groupby("Card on Dark Web").sum().plot.bar(title="Fraudes segun disponibilidad en dark web")
# fig = plot.get_figure()
# fig.savefig("./by_dark_web.svg")

# histograma de cantitad de fraudes
# segun ultimo cambio de clave
last_changed = joined[["Year PIN last Changed", "Fraud Count"]]
last_changed = last_changed.loc[last_changed["Fraud Count"]!=0]
last_changed = last_changed.loc[last_changed.index.repeat(last_changed["Fraud Count"].max(0))]
plot = last_changed.plot.hist(column="Year PIN last Changed", bins=18, title="Cantidad de fraudes por año de última renovacion")
fig = plot.get_figure()
fig.savefig("./by_pin_change.svg")
# *podria ser interesante filtrar segun si la transaccion
# se hizo antes o despues del cambio, pero no tuve tiempo

#plot = joined.plot.hist(column="Year PIN last Changed", bins=18, title="Cantidad de última renovacion por año")
#fig = plot.get_figure()
#fig.savefig("./renew_by_year.svg")

print("Listo! :)")
