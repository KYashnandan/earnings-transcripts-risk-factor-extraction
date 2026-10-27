"""Risk phrases per category. Matched on lemmas, so 'regulators' and 'investigating' also hit."""
LEXICON = {
 "regulatory": ["regulation","regulatory","regulator","antitrust","investigation","investigate","subpoena","lawsuit","litigation","legal proceeding","sanction","tariff","penalty"],
 "supply_chain": ["supply chain disruption","supply chain constraint","supply chain challenge","supply chain issue","supply constraint","shortage","shipping delay","port congestion","logistics challenge","production disruption","bottleneck","lead time"],
 "macro": ["inflation","inflationary pressure","recession","interest rate","headwind","slowdown","geopolitical","uncertainty","volatility","economic weakness"],
 "competition": ["competition","competitor","competitive pressure","pricing pressure","market share loss","price war","new entrant"],
 "liquidity": ["liquidity","refinancing","covenant","cash burn","credit loss","default","downgrade","impairment","write-down","going concern","leverage"],
 "cyber": ["cyber","cyberattack","data breach","security incident","ransomware","outage","breach"],
}
