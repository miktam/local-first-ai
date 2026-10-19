# exp_044: 30 test leads for Andrei's read

Check each label against the message and LABELS.md. Mark any lead you would label differently.

**X6-10** (sv, form, standard)

> Hej, vi är intresserade av objekt 4521 (lägenheten i Puerto Banús). Vi är fyra vuxna och vill hyra 10–25 juli 2027. Vad kostar det per vecka och ingår städning? Mvh Lena Sjögren

intent `rent_holiday` · type `apartment` · timeline `later` · financing `None` · foreign `None` · qualified `True` · urgency `2` · action `send_listings` · budget `None`

**X4-11** (es, whatsapp, hard_budget)

> Hola, buscamos villa en Benahavís o La Quinta, 4 dorm, vistas al mar, presupuesto hasta un millón y medio. Venimos a ver casas la primera semana de noviembre. Somos de Barcelona, no necesitamos financiación. Gracias! Marc

intent `buy` · type `villa` · timeline `now` · financing `False` · foreign `False` · qualified `True` · urgency `2` · action `send_listings` · budget `1500000`

**X5-25** (de, whatsapp, standard)

> Hallo Frau García, wir haben am Dienstag mit Ihnen die Wohnung in Cancelada besichtigt (die mit der großen Terrasse). Wir möchten ein Angebot machen: 385.000 €, die Finanzierung über unsere Bank steht bereits. Bitte rufen Sie mich heute noch an: +34 600 000 104. Danke, Sabine Kraus

intent `buy` · type `apartment` · timeline `now` · financing `True` · foreign `None` · qualified `True` · urgency `3` · action `call_now` · budget `385000`

**X3-36** (pl, email, standard)

> Dzień dobry,

szukamy apartamentu na ferie zimowe: 10 dni w lutym (dokładnie 13–23.02), rodzina 2+2, w Marbelli lub okolicach, 2 sypialnie, ogrzewanie koniecznie 🙂 Budżet do 2.500 € za cały pobyt.

Czy mają Państwo coś w ofercie?

Pozdrawiam,
Magda

intent `rent_holiday` · type `apartment` · timeline `later` · financing `None` · foreign `None` · qualified `True` · urgency `0` · action `send_listings` · budget `2500`

**X3-41** (pl, voice, standard)

> Dzień dobry, no więc tak, odziedziczyliśmy z bratem i siostrą dom po rodzicach w Mijas, w Mijas Pueblo właściwie, taki yyy, stary dom, chyba ze trzy sypialnie, z tarasem, no i chcemy go sprzedać, bo nikt z nas tam nie mieszka, wszyscy jesteśmy w Polsce. Nie ma pośpiechu, ale chcielibyśmy to załatwić jeszcze w tym roku, żeby nie przeciągać spadku. Ile taki dom może być teraz wart? I czy możecie to poprowadzić, jak nas nie ma na miejscu? To ja czekam na odpowiedź, dziękuję.

intent `sell` · type `villa` · timeline `months` · financing `None` · foreign `True` · qualified `True` · urgency `1` · action `book_valuation` · budget `None`

**X4-21** (es, email, standard)

> Estimado propietario, su anuncio ha sido seleccionado para nuestro programa de compradores internacionales. Un inversor de Dubái está dispuesto a pagar por encima del precio de mercado. Para reservar la cita abone la tasa de gestión de 149 € en el enlace: http://inmo-premium-ofertas.example.com/pago. Atentamente, Dpto. Internacional

intent `spam` · type `unspecified` · timeline `unspecified` · financing `None` · foreign `None` · qualified `False` · urgency `0` · action `review` · budget `None`

**X4-04** (es, email, hard_b2b)

> Estimados señores, Somos Iberia Mobility Partners, empresa de relocation. Gestionamos el traslado a Málaga de 6 ingenieros de un cliente tecnológico que se incorporan en su sede de Málaga capital a mediados de diciembre. Necesitamos 6 viviendas de alquiler de larga duración (mínimo 12 meses), 1–2 dormitorios, amuebladas, en Málaga centro, Teatinos o Torremolinos, con presupuesto de hasta 1.600 €/mes por vivienda. El contrato lo firmaría la empresa como arrendataria. ¿Podrían enviarnos disponibilidad esta semana? Quedamos a su disposición. Cristina Vidal, Account Manager, Iberia Mobility Partners

intent `rent_long` · type `None` · timeline `months` · financing `None` · foreign `None` · qualified `True` · urgency `1` · action `send_listings` · budget `1600`

**X4-57** (es, voice, hard_holiday)

> Hola, eh, sí, mira, somos de Pamplona, venimos una semana cada dos meses más o menos a Benalmádena, siempre a alquilar… y ya, eh, ya estamos cansados de pagar alquileres, queremos algo nuestro para las escapadas, para los veranos y Semana Santa… un apartamento de dos habitaciones, con piscina, en Benalmádena Costa o Arroyo de la Miel, que no pase de 220.000, vamos, hipoteca una parte… venimos el puente de noviembre, del 31 al 2, podríamos ver cosas… llámame al +34 600 000 926, soy Javier.

intent `buy` · type `apartment` · timeline `now` · financing `True` · foreign `False` · qualified `True` · urgency `2` · action `call_now` · budget `220000`

**X7-31** (fr, form, standard)

> Bonjour, nous déménageons à Marbella en septembre de l'année prochaine (mutation professionnelle de mon mari). Nous cherchons une maison de ville avec 3 chambres, proche d'une école internationale, idéalement San Pedro ou Nueva Andalucía. Budget 700 000 € maximum, nous aurons besoin d'un prêt pour une partie. Nous viendrons probablement visiter en février.

intent `buy` · type `townhouse` · timeline `later` · financing `True` · foreign `disagree` · qualified `True` · urgency `0` · action `send_listings` · budget `700000`

**X4-27** (es, email, hard_b2b)

> Estimados, Somos Promociones Guadiaro SL, promotora con 3 obras entregadas en Manilva y Estepona. Buscamos suelo urbano o urbanizable finalista en Estepona, Casares Costa o Manilva para 20–40 viviendas, con una inversión en suelo de hasta 4 M€. Compramos con fondos propios y podemos cerrar en 60 días si la parcela tiene licencia o está cerca de tenerla. ¿Tienen suelo en cartera o conocen propietarios que vendan? Atentamente, Javier Mena, Director de Expansión

intent `invest` · type `plot` · timeline `months` · financing `False` · foreign `disagree` · qualified `True` · urgency `1` · action `send_listings` · budget `4000000`

**X1-03** (en, email, standard)

> Dear agency,

We are thinking of selling our 3-bedroom townhouse in Riviera del Sol and would like to list it with you. Before we do, could you tell us roughly what it is worth? We have no idea what the current market is like. We'd like it on the market within the next couple of weeks.

Regards,
Susan and Clive Barnett

intent `sell` · type `townhouse` · timeline `now` · financing `None` · foreign `None` · qualified `True` · urgency `2` · action `book_valuation` · budget `None`

**X5-02** (de, email, hard_holiday)

> Guten Tag,

wir sind gerade eine Woche hier in Estepona im Urlaub und haben uns in die Gegend verliebt. Wir möchten eine Wohnung mit Meerblick kaufen, die wir für unsere Urlaube nutzen und sonst leer stehen lassen. 2 Schlafzimmer, Budget 650.000 €, Eigenkapital vorhanden, keine Finanzierung nötig. Wir sind noch bis Samstag hier und würden gern diese Woche noch etwas besichtigen. Rufen Sie uns gern an.

Grüße, Familie Brandt aus Köln

intent `buy` · type `apartment` · timeline `now` · financing `False` · foreign `True` · qualified `True` · urgency `3` · action `call_now` · budget `650000`

**X2-22** (en, whatsapp, standard)

> need a place for 10 days from 14 nov, 2 adults 1 baby, benalmadena or torremolinos, walking to beach, max 1500 total, pls reply quick its a bit last minute i know lol

intent `rent_holiday` · type `unspecified` · timeline `now` · financing `None` · foreign `None` · qualified `True` · urgency `2` · action `send_listings` · budget `1500`

**X2-37** (en, email, standard)

> Dear team, we are a family of four from Dublin planning a permanent move to the costa del sol in 2027. We are looking for a 4 bed villa with a garden close to an international school, Sotogrande or the San Pedro / Guadalmina area, budget 900k to 1.1M, we would need a mortgage for about 40%. We will be in Spain on a scouting trip the first week of december and would love to line up 5-6 viewings. Many thanks, Ciara and Tom

intent `buy` · type `villa` · timeline `months` · financing `True` · foreign `True` · qualified `True` · urgency `1` · action `send_listings` · budget `1100000`

**X3-45** (pl, email, hard_b2b)

> Dzień dobry,

prowadzimy biuro nieruchomości CostaPol Estates w Fuengiroli. Mamy w tym roku kilkudziesięciu klientów z Polski szukających głównie w Marbelli i Esteponie, gdzie nie mamy własnych ofert. Proponujemy stałą współpracę: dzielimy prowizję 50/50 za każdą transakcję z naszym klientem na Państwa ofertach. Możemy też wymieniać się ofertami na wyłączność.

Czy spotkamy się w przyszłym tygodniu w Państwa biurze?

Pozdrawiam,
Michał Król
CostaPol Estates

intent `spam` · type `unspecified` · timeline `unspecified` · financing `None` · foreign `None` · qualified `False` · urgency `0` · action `review` · budget `None`

**X7-41** (fr, whatsapp, standard)

> slt on cherche un appart pour la semaine prochaine, 7 nuits a partir de samedi, a estepona ou manilva 4 pers budget 900 la semaine merci 👍 +34 600 000 173

intent `rent_holiday` · type `apartment` · timeline `now` · financing `None` · foreign `None` · qualified `True` · urgency `2` · action `call_now` · budget `900`

**X8-32** (ru, email, hard_budget)

> Здравствуйте. Переезжаю в Фуэнхиролу по работе, нужна квартира на длительный срок, минимум год. Две спальни, с мебелью, парковка, желательно ближе к центру и поезду. Могу платить две с половиной тысячи в месяц, не больше. Заезд в декабре. Подскажите, что есть, и какие документы нужны для договора. Руслан

intent `rent_long` · type `apartment` · timeline `months` · financing `None` · foreign `None` · qualified `True` · urgency `1` · action `send_listings` · budget `2500`

**X7-38** (fr, form, hard_budget)

> Bonjour, nous cherchons une villa à louer à l'année à Nueva Andalucía ou Guadalmina, 4 chambres, piscine, à partir de janvier, bail de 12 mois renouvelable. Nous avons deux enfants scolarisés à Aloha College. Notre budget est de deux mille cinq cents euros par mois, charges non comprises. Merci de nous envoyer vos disponibilités.

intent `rent_long` · type `villa` · timeline `None` · financing `None` · foreign `None` · qualified `True` · urgency `None` · action `send_listings` · budget `2500`

**X6-72** (sv, whatsapp, standard)

> hej!! finns det nån lägenhet ledig i la cala eller calahonda nu i november, 2 veckor, 2 vuxna? flexibla med datum, max 900 euro totalt 🙏

intent `rent_holiday` · type `apartment` · timeline `now` · financing `None` · foreign `None` · qualified `True` · urgency `2` · action `send_listings` · budget `900`

**X1-62** (en, whatsapp, standard)

> Hi are you hiring agents? I have 5 years experience selling in Mallorca and I'm moving to Marbella next month, fluent Spanish & English, own car. CV on request 🙂

intent `spam` · type `unspecified` · timeline `unspecified` · financing `None` · foreign `None` · qualified `False` · urgency `0` · action `review` · budget `None`

**X3-51** (pl, email, standard)

> Dzień dobry,

Portal DomyNaSłońcu.pl to największy polski serwis z nieruchomościami w Hiszpanii (120 tys. odwiedzin miesięcznie). Oferujemy pakiet Premium dla biur: 50 ogłoszeń, wyróżnienie na stronie głównej, newsletter do 30 tys. subskrybentów – 490 € za kwartał. Do końca października drugi kwartał gratis.

Chętnie założę Państwu konto testowe.

Pozdrawiam,
Ola z DomyNaSłońcu.pl

intent `spam` · type `unspecified` · timeline `unspecified` · financing `None` · foreign `None` · qualified `False` · urgency `0` · action `review` · budget `None`

**X1-41** (en, whatsapp, hard_b2b)

> Hi 👋 I'm a property photographer & home stager based in Marbella, I do photos, drone video and virtual staging for agencies. Intro offer: first shoot free so you can see the quality. Portfolio on request. Can I come by the office this week?

intent `spam` · type `unspecified` · timeline `unspecified` · financing `None` · foreign `None` · qualified `False` · urgency `0` · action `review` · budget `None`

**X2-57** (en, email, standard)

> Hello, I am Sofia from Global Investor Network. We have 2,000+ HNW investors looking at spanish real estate. For only €990/year your listings get promoted to our network. Limited places for Marbella agencies this quarter. Reply YES for the brochure.

intent `spam` · type `unspecified` · timeline `unspecified` · financing `None` · foreign `None` · qualified `False` · urgency `0` · action `review` · budget `None`

**X7-42** (fr, email, hard_b2b)

> Bonjour,

Nous générons chaque mois plus de 300 leads acheteurs qualifiés (français, belges, scandinaves) pour des agences de la Costa del Sol via Meta Ads et des landing pages dédiées. Nos partenaires paient uniquement au lead livré, à partir de 35 € le contact. Puis-je vous envoyer nos résultats du dernier trimestre et vous proposer un test sur 50 leads ?

Sébastien Roux
LeadsCosta

intent `spam` · type `unspecified` · timeline `unspecified` · financing `None` · foreign `None` · qualified `False` · urgency `0` · action `review` · budget `None`

**X7-48** (fr, whatsapp, hard_holiday)

> Bonjour, nous voulons bloquer une maison pour juillet, 3 semaines du 5 au 26, Nueva Andalucía ou Benahavís, 10 personnes, piscine. On a 12 000 € pour le séjour et on peut verser un acompte tout de suite pour sécuriser. C'est pour cet été.

intent `rent_holiday` · type `villa` · timeline `None` · financing `None` · foreign `None` · qualified `True` · urgency `1` · action `send_listings` · budget `12000`

**X5-01** (de, whatsapp, standard)

> hallo wir suchen ne wohnung in nueva andalucia 2-3 schlafzimmer bis 800k sind nächste woche vor ort 😊 können sie uns was zeigen? 0034 600 000 101 lg petra u. jochen

intent `buy` · type `apartment` · timeline `now` · financing `None` · foreign `None` · qualified `True` · urgency `2` · action `call_now` · budget `800000`

**X2-59** (en, email, standard)

> Hello, I am moving to Málaga from Portugal in november to start at the hospital (nurse). I need a studio or 1 bed near Carlos Haya / Teatinos, max 900€ per month, furnished, from 15 nov. Contact: ines.m@example.com. Thank you, Inês

intent `rent_long` · type `apartment` · timeline `now` · financing `None` · foreign `True` · qualified `True` · urgency `2` · action `send_listings` · budget `900`

**X2-69** (en, whatsapp, standard)

> Hola! need more 5⭐ reviews on google? we deliver 50 verified reviews in 7 days, 100% safe, agencies in marbella already using us, whatsapp back for prices 🔥

intent `spam` · type `unspecified` · timeline `unspecified` · financing `None` · foreign `None` · qualified `False` · urgency `0` · action `review` · budget `None`

**X7-73** (fr, email, hard_holiday)

> Bonjour,

Nous voulons nous assurer une maison pour Noël et Nouvel An, du 20 décembre au 3 janvier, sur Marbella ou Elviria : 4 chambres, chauffage, pour 8 personnes dont des grands-parents. Nous sommes prêts à payer d'avance pour la bloquer. Budget 6 000 € pour les deux semaines, au-delà nous ne suivrons pas.

Merci,
Famille Garnier

intent `rent_holiday` · type `villa` · timeline `None` · financing `None` · foreign `None` · qualified `True` · urgency `None` · action `send_listings` · budget `6000`

**X5-27** (de, whatsapp, standard)

> hallo, eine freundin hat mir von manilva erzählt. wie ist la duquesa so für familien, gibt es schulen in der nähe und ist es im winter sehr ruhig? danke für eine kurze einschätzung

intent `info` · type `unspecified` · timeline `unspecified` · financing `None` · foreign `None` · qualified `False` · urgency `0` · action `answer_question` · budget `None`

