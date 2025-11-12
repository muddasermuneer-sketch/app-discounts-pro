import streamlit as st
import pandas as pd
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any, Set
from datetime import date, time, datetime

st.set_page_config(page_title="D365 FO Retail Discounts Pro", page_icon="🏷️", layout="wide")

# ---------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------
@dataclass
class Store:
    name: str
    store_groups: List[str] = field(default_factory=list)     # hierarchy groups
    price_groups: List[str] = field(default_factory=list)     # assigned price groups

@dataclass
class CustomerContext:
    is_employee: bool = False
    affiliations: List[str] = field(default_factory=list)     # e.g., Military, Senior
    price_groups: List[str] = field(default_factory=list)     # customer price groups

@dataclass
class CartLine:
    sku: str
    name: str
    category: str
    unit_price: float
    qty: int

@dataclass
class DiscountRule:
    name: str
    rule_type: str  # 'Simple %', 'Simple amount', 'Threshold %', 'Threshold amount', 'Mix&Match BxGy'
    percent: float = 0.0
    amount: float = 0.0
    threshold_amount: float = 0.0
    buy_qty: int = 0
    get_qty: int = 0
    scope: str = "Line"  # 'Line' or 'Order'
    # Filters
    category_filter: Optional[str] = None
    sku_filter: Optional[str] = None
    require_employee: bool = False
    required_affiliations: List[str] = field(default_factory=list)  # any-of affiliations required
    coupon_code: Optional[str] = None
    channels: List[str] = field(default_factory=list)          # explicit channel names (stores)
    store_groups: List[str] = field(default_factory=list)      # hierarchy nodes
    price_groups: List[str] = field(default_factory=list)      # linked price groups
    concurrency: str = "Best price"        # 'Exclusive', 'Best price', 'Compounded'
    priority: int = 10
    # Validation period
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    valid_days: List[str] = field(default_factory=list)        # ["Mon","Tue",...]
    start_time: Optional[time] = None
    end_time: Optional[time] = None

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
DAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

def parse_csv_list(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return [s.strip() for s in text.split(",") if s.strip()]

def line_subtotal(line: CartLine) -> float:
    return line.unit_price * line.qty

def compute_simple_percent(line: CartLine, percent: float) -> float:
    return line_subtotal(line) * max(percent, 0.0) / 100.0

def compute_simple_amount(line: CartLine, amount: float) -> float:
    return min(line_subtotal(line), max(amount, 0.0))

def apply_mix_match(cart: List[CartLine], indices: List[int], buy_qty: int, get_qty: int, percent_off: float) -> Dict[int, float]:
    if buy_qty <= 0 or get_qty <= 0 or percent_off <= 0:
        return {}
    units: List[tuple] = []
    for idx in indices:
        line = cart[idx]
        for _ in range(line.qty):
            units.append((line.unit_price, idx))
    if not units:
        return {}
    units.sort(key=lambda x: x[0])
    bundle_size = buy_qty + get_qty
    num_bundles = len(units) // bundle_size
    discount_units_needed = num_bundles * get_qty
    per_line_discount: Dict[int, float] = {}
    for price, idx in units[:discount_units_needed]:
        discount_value = price * percent_off / 100.0
        per_line_discount[idx] = per_line_discount.get(idx, 0.0) + discount_value
    return per_line_discount

def apply_threshold(total: float, threshold_amount: float) -> bool:
    return total >= threshold_amount

def format_money(x: float) -> str:
    return f"{x:,.2f}"

def current_day_time(now: datetime):
    return DAYS[now.weekday()], now.time()

# Eligibility checks for rule
def rule_active_now(rule: DiscountRule, today: date, now_time: time, day_code: str) -> bool:
    if rule.start_date and today < rule.start_date:
        return False
    if rule.end_date and today > rule.end_date:
        return False
    if rule.valid_days and day_code not in rule.valid_days:
        return False
    if rule.start_time and now_time < rule.start_time:
        return False
    if rule.end_time and now_time > rule.end_time:
        return False
    return True

def store_matches_rule(rule: DiscountRule, store: Store) -> bool:
    # If channels specified, store name must be included
    if rule.channels and store.name not in rule.channels:
        return False
    # If rule store_groups specified, must intersect with store.store_groups
    if rule.store_groups:
        if not set(rule.store_groups).intersection(store.store_groups):
            return False
    return True

def price_group_link_ok(rule: DiscountRule, store: Store, customer: CustomerContext) -> bool:
    # If rule has price groups, they must appear in EITHER store or customer price groups (like D365 linkage)
    if not rule.price_groups:
        return True
    pool: Set[str] = set(store.price_groups) | set(customer.price_groups)
    return bool(set(rule.price_groups).intersection(pool))

def customer_matches_rule(rule: DiscountRule, customer: CustomerContext) -> bool:
    if rule.require_employee and not customer.is_employee:
        return False
    if rule.required_affiliations:
        if not set(rule.required_affiliations).intersection(customer.affiliations):
            return False
    return True

def line_filters_ok(rule: DiscountRule, line: CartLine) -> bool:
    if rule.category_filter and rule.category_filter != line.category:
        return False
    if rule.sku_filter:
        lst = set(rule.sku_filter)
        if line.sku not in lst:
            return False
    return True

# ---------------------------------------------------------------------
# Discount engine
# ---------------------------------------------------------------------
def evaluate_discounts(cart: List[CartLine], rules: List[DiscountRule], store: Store, customer: CustomerContext,
                       today: date, coupon: Optional[str], now: Optional[datetime]=None) -> Dict[str, Any]:

    now = now or datetime.now()
    day_code, now_time = current_day_time(now)

    line_totals = [line_subtotal(l) for l in cart]
    base_total = sum(line_totals)

    per_line_best: List[float] = [0.0 for _ in cart]
    per_line_stack: List[float] = [0.0 for _ in cart]
    exclusive_applied = False
    exclusive_name: Optional[str] = None
    audit_log: List[str] = []

    rules_sorted = sorted(rules, key=lambda r: (r.priority, r.name.lower()))

    for rule in rules_sorted:
        if exclusive_applied:
            break

        # Global-level checks
        if not rule_active_now(rule, today, now_time, day_code):
            continue
        if not store_matches_rule(rule, store):
            continue
        if not price_group_link_ok(rule, store, customer):
            continue
        if not customer_matches_rule(rule, customer):
            continue
        if rule.coupon_code and (coupon or "") != rule.coupon_code:
            continue

        # Line eligibility
        eligible_indices = [i for i, ln in enumerate(cart) if line_filters_ok(rule, ln)]
        if not eligible_indices and rule.rule_type not in ("Threshold %", "Threshold amount"):
            continue

        per_line_candidate: Dict[int, float] = {}

        if rule.rule_type == "Simple %":
            for idx in eligible_indices:
                per_line_candidate[idx] = compute_simple_percent(cart[idx], rule.percent)

        elif rule.rule_type == "Simple amount":
            for idx in eligible_indices:
                per_line_candidate[idx] = compute_simple_amount(cart[idx], rule.amount)

        elif rule.rule_type == "Mix&Match BxGy":
            per_line_candidate = apply_mix_match(cart, eligible_indices, rule.buy_qty, rule.get_qty, rule.percent)

        elif rule.rule_type in ("Threshold %", "Threshold amount"):
            eligible_amount = sum(line_totals[i] for i in eligible_indices) if rule.scope == "Line" else base_total
            if not apply_threshold(eligible_amount, rule.threshold_amount):
                continue
            if rule.rule_type == "Threshold amount":
                target_amount = rule.amount
                denom = sum(line_totals[i] for i in eligible_indices) if rule.scope == "Line" else base_total
                if denom <= 0:
                    continue
                for idx in (eligible_indices if rule.scope == "Line" else range(len(cart))):
                    weight = line_totals[idx] / denom
                    per_line_candidate[idx] = min(line_totals[idx], target_amount * weight)
            else:
                pct = rule.percent
                for idx in (eligible_indices if rule.scope == "Line" else range(len(cart))):
                    per_line_candidate[idx] = line_totals[idx] * pct / 100.0

        if not per_line_candidate:
            continue

        if rule.concurrency == "Exclusive":
            per_line_best = [0.0 for _ in cart]
            per_line_stack = [0.0 for _ in cart]
            for idx, disc in per_line_candidate.items():
                cap = max(0.0, line_totals[idx] - per_line_stack[idx])
                per_line_stack[idx] = min(disc, cap)
            exclusive_applied = True
            exclusive_name = rule.name
            audit_log.append(f"Exclusive → {rule.name}")
            break

        elif rule.concurrency == "Compounded":
            for idx, disc in per_line_candidate.items():
                cap = max(0.0, line_totals[idx] - per_line_stack[idx])
                applied = min(disc, cap)
                per_line_stack[idx] += applied
            audit_log.append(f"Compounded → {rule.name}")

        else:
            for idx, disc in per_line_candidate.items():
                per_line_best[idx] = max(per_line_best[idx], disc)
            audit_log.append(f"Best price candidate → {rule.name}")

    per_line_final = [max(best, 0.0) + stack for best, stack in zip(per_line_best, per_line_stack)]
    for i in range(len(per_line_final)):
        per_line_final[i] = min(per_line_final[i], line_totals[i])

    total_discount = sum(per_line_final)
    final_total = base_total - total_discount

    breakdown = []
    for i, line in enumerate(cart):
        breakdown.append({
            "SKU": line.sku,
            "Name": line.name,
            "Category": line.category,
            "Qty": line.qty,
            "Unit Price": line.unit_price,
            "Line Subtotal": line_totals[i],
            "Discount Applied": per_line_final[i],
            "Net Line": line_totals[i] - per_line_final[i],
        })

    return {
        "base_total": base_total,
        "total_discount": total_discount,
        "final_total": final_total,
        "breakdown": breakdown,
        "audit": audit_log,
        "exclusive_name": exclusive_name
    }

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("🏷️ D365 FO Retail Discounts — Pro Simulator")
st.caption("Adds Price Groups, Store Hierarchies, and Validation Periods to the earlier app.")

# Session state
if "rules_pro" not in st.session_state:
    st.session_state.rules_pro: List[DiscountRule] = []
if "cart_pro" not in st.session_state:
    st.session_state.cart_pro: List[CartLine] = []
if "stores" not in st.session_state:
    st.session_state.stores: Dict[str, Store] = {}
if "customer_ctx" not in st.session_state:
    st.session_state.customer_ctx = CustomerContext()

# Top rows
c0, c1, c2, c3 = st.columns([1,1,1,1])
with c0:
    store_name = st.selectbox("Store / Channel", options=list(st.session_state.stores.keys()) or ["Default"], index=0)
with c1:
    coupon = st.text_input("Coupon (optional)", value="")
with c2:
    today = st.date_input("Pricing Date", value=date.today())
with c3:
    if st.button("Load demo data (Pro)"):
        # Stores and groups
        st.session_state.stores = {
            "Store-001": Store(name="Store-001", store_groups=["Dubai", "UAE"], price_groups=["Retail", "MallEmp"]),
            "Store-002": Store(name="Store-002", store_groups=["AbuDhabi", "UAE"], price_groups=["Retail"]),
        }
        # Customer
        st.session_state.customer_ctx = CustomerContext(is_employee=False, affiliations=["Senior"], price_groups=["LoyaltyGold"])
        # Rules
        st.session_state.rules_pro = [
            DiscountRule(
                name="Mall Employees 10%",
                rule_type="Simple %",
                percent=10,
                scope="Line",
                required_affiliations=[],
                require_employee=True,
                store_groups=["Dubai"],
                price_groups=["MallEmp"],
                concurrency="Compounded",
                priority=5,
                valid_days=["Fri","Sat"],
            ),
            DiscountRule(
                name="Loyalty Gold 5%",
                rule_type="Simple %",
                percent=5,
                scope="Line",
                price_groups=["LoyaltyGold"],
                concurrency="Best price",
                priority=20,
            ),
            DiscountRule(
                name="Weekend Basket 20 off 150",
                rule_type="Threshold amount",
                amount=20,
                threshold_amount=150,
                scope="Order",
                channels=["Store-001","Store-002"],
                concurrency="Best price",
                priority=50,
                valid_days=["Fri","Sat"],
                start_time=time(9,0),
                end_time=time(23,0),
            ),
            DiscountRule(
                name="Buy2Get1 Snacks",
                rule_type="Mix&Match BxGy",
                percent=100,
                buy_qty=2,
                get_qty=1,
                scope="Line",
                category_filter="Snacks",
                price_groups=["Retail"],
                concurrency="Best price",
                priority=10,
            ),
            DiscountRule(
                name="Coupon SAVE25",
                rule_type="Threshold amount",
                amount=25,
                threshold_amount=200,
                scope="Order",
                coupon_code="SAVE25",
                concurrency="Exclusive",
                priority=1,
            ),
        ]
        # Cart
        st.session_state.cart_pro = [
            CartLine("SKU-100", "Cola 1.5L", "Beverages", 2.50, 10),
            CartLine("SKU-200", "Potato Chips", "Snacks", 1.75, 6),
            CartLine("SKU-300", "Chocolate Bar", "Snacks", 1.25, 3),
            CartLine("SKU-400", "Basmati Rice 5kg", "Grocery", 12.00, 2),
        ]
        st.success("Demo (Pro) loaded.")

st.divider()

tab_masters, tab_rules, tab_cart, tab_customer, tab_calc, tab_help = st.tabs(
    ["Master Data", "Discount Rules", "Cart", "Customer", "Calculate", "Help"]
)

# ----------------------------- Master Data ------------------------------
with tab_masters:
    st.subheader("Stores & Groups")
    with st.form("store_form", clear_on_submit=True):
        c = st.columns(4)
        sname = c[0].text_input("Store name*", value="")
        sgroups = c[1].text_input("Store groups (comma)", value="")
        spgroups = c[2].text_input("Price groups (comma)", value="")
        add_store = c[3].form_submit_button("Add/Update store", use_container_width=True)
        if add_store and sname:
            st.session_state.stores[sname] = Store(
                name=sname,
                store_groups=parse_csv_list(sgroups),
                price_groups=parse_csv_list(spgroups),
            )
            st.success(f"Store '{sname}' saved.")
    if st.session_state.stores:
        df = pd.DataFrame([asdict(v) for v in st.session_state.stores.values()])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No stores yet. Add one or load demo.")

# ----------------------------- Rules Tab --------------------------------
with tab_rules:
    st.subheader("Discount Rules (with Price Groups & Hierarchies)")
    with st.form("rule_form_pro", clear_on_submit=True):
        col = st.columns(4)
        name = col[0].text_input("Rule name*", value="")
        rule_type = col[1].selectbox("Type*", ["Simple %", "Simple amount", "Threshold %", "Threshold amount", "Mix&Match BxGy"])
        concurrency = col[2].selectbox("Concurrency", ["Best price", "Compounded", "Exclusive"], index=0)
        priority = col[3].number_input("Priority (lower first)", value=10, step=1)

        col2 = st.columns(4)
        percent = col2[0].number_input("% (if applicable)", value=0.0, step=0.1)
        amount = col2[1].number_input("Fixed amount (if applicable)", value=0.0, step=0.1)
        threshold_amount = col2[2].number_input("Threshold amount (if applicable)", value=0.0, step=0.1)
        scope = col2[3].selectbox("Scope", ["Line", "Order"], index=0)

        col3 = st.columns(4)
        buy_qty = col3[0].number_input("Buy Qty (BxGy)", value=0, step=1)
        get_qty = col3[1].number_input("Get Qty (BxGy)", value=0, step=1)
        category_filter = col3[2].text_input("Category filter (exact match)", value="")
        sku_filter = col3[3].text_input("SKU filter (comma-separated)", value="")

        col4 = st.columns(4)
        require_employee = col4[0].checkbox("Employee required?")
        required_affiliations = col4[1].text_input("Required affiliations (any-of, comma)", value="")
        coupon_code = col4[2].text_input("Coupon code", value="")
        channels = col4[3].text_input("Target stores (comma)", value="")

        col5 = st.columns(4)
        store_groups = col5[0].text_input("Target store groups (comma)", value="")
        price_groups = col5[1].text_input("Linked price groups (comma)", value="")
        valid_days = col5[2].multiselect("Valid days", DAYS, default=[])
        start_date = col5[3].date_input("Start date", value=None)

        col6 = st.columns(4)
        end_date = col6[0].date_input("End date", value=None)
        start_time = col6[1].time_input("Start time", value=None)
        end_time = col6[2].time_input("End time", value=None)
        submitted = col6[3].form_submit_button("Add rule", use_container_width=True)

        if submitted:
            st.session_state.rules_pro.append(
                DiscountRule(
                    name=name or f"Rule {len(st.session_state.rules_pro)+1}",
                    rule_type=rule_type,
                    percent=float(percent),
                    amount=float(amount),
                    threshold_amount=float(threshold_amount),
                    buy_qty=int(buy_qty),
                    get_qty=int(get_qty),
                    scope=scope,
                    category_filter=category_filter or None,
                    sku_filter=parse_csv_list(sku_filter) or None,
                    require_employee=bool(require_employee),
                    required_affiliations=parse_csv_list(required_affiliations),
                    coupon_code=coupon_code or None,
                    channels=parse_csv_list(channels),
                    store_groups=parse_csv_list(store_groups),
                    price_groups=parse_csv_list(price_groups),
                    concurrency=concurrency,
                    priority=int(priority),
                    start_date=start_date,
                    end_date=end_date,
                    valid_days=list(valid_days),
                    start_time=start_time,
                    end_time=end_time,
                )
            )
            st.success("Rule added.")
    if st.session_state.rules_pro:
        df_rules = pd.DataFrame([asdict(r) for r in st.session_state.rules_pro])
        st.dataframe(df_rules, use_container_width=True, hide_index=True)
        idx_to_del = st.number_input("Delete rule at index (1-based)", min_value=0, value=0, step=1)
        if st.button("Delete rule"):
            if 0 < idx_to_del <= len(st.session_state.rules_pro):
                st.session_state.rules_pro.pop(idx_to_del-1)
                st.toast("Rule deleted.", icon="🗑️")
            else:
                st.warning("Provide a valid index to delete.")
        st.download_button(
            "Download rules (CSV)",
            data=df_rules.to_csv(index=False).encode("utf-8"),
            file_name="discount_rules_pro.csv",
            mime="text/csv",
        )
    else:
        st.info("No rules yet. Add or load demo.")

# ----------------------------- Cart Tab ---------------------------------
with tab_cart:
    st.subheader("Cart Lines")
    with st.form("cart_form_pro", clear_on_submit=True):
        c = st.columns(5)
        sku = c[0].text_input("SKU*", value="")
        name = c[1].text_input("Name*", value="")
        category = c[2].text_input("Category*", value="")
        unit_price = c[3].number_input("Unit price*", value=0.0, step=0.01, min_value=0.0, format="%.4f")
        qty = c[4].number_input("Qty*", value=1, step=1, min_value=1)
        add = st.form_submit_button("Add line", use_container_width=True)
        if add:
            st.session_state.cart_pro.append(CartLine(sku, name, category, float(unit_price), int(qty)))
            st.success("Line added.")
    if st.session_state.cart_pro:
        df_cart = pd.DataFrame([asdict(x) for x in st.session_state.cart_pro])
        st.dataframe(df_cart, use_container_width=True, hide_index=True)
        idx_to_del2 = st.number_input("Delete line at index (1-based)", min_value=0, value=0, step=1, key="del_line_pro")
        if st.button("Delete line"):
            if 0 < idx_to_del2 <= len(st.session_state.cart_pro):
                st.session_state.cart_pro.pop(idx_to_del2-1)
                st.toast("Line deleted.", icon="🗑️")
            else:
                st.warning("Provide a valid index to delete.")
        st.download_button(
            "Download cart (CSV)",
            data=df_cart.to_csv(index=False).encode("utf-8"),
            file_name="cart_lines_pro.csv",
            mime="text/csv",
        )
    else:
        st.info("Cart is empty. Add items or load demo.")

# ----------------------------- Customer Tab -----------------------------
with tab_customer:
    st.subheader("Customer Context")
    c = st.columns(4)
    st.session_state.customer_ctx.is_employee = c[0].checkbox("Employee?", value=st.session_state.customer_ctx.is_employee)
    aff_in = c[1].text_input("Affiliations (comma)", value=",".join(st.session_state.customer_ctx.affiliations))
    pg_in = c[2].text_input("Customer price groups (comma)", value=",".join(st.session_state.customer_ctx.price_groups))
    save_cust = c[3].button("Save customer context")
    if save_cust:
        st.session_state.customer_ctx.affiliations = parse_csv_list(aff_in)
        st.session_state.customer_ctx.price_groups = parse_csv_list(pg_in)
        st.success("Customer context saved.")

# ----------------------------- Calculate Tab ----------------------------
with tab_calc:
    st.subheader("Price & Discount Calculation")
    if not st.session_state.cart_pro:
        st.warning("Add cart lines first.")
    else:
        # Resolve selected store or default
        if store_name not in st.session_state.stores:
            store = Store(name=store_name, store_groups=[], price_groups=[])
        else:
            store = st.session_state.stores[store_name]
        result = evaluate_discounts(
            cart=st.session_state.cart_pro,
            rules=st.session_state.rules_pro,
            store=store,
            customer=st.session_state.customer_ctx,
            today=today,
            coupon=coupon.strip() or None,
        )
        colA, colB, colC = st.columns(3)
        colA.metric("Base total", format_money(result["base_total"]))
        colB.metric("Total discount", f"-{format_money(result['total_discount'])}")
        colC.metric("Final total", format_money(result["final_total"]))

        st.markdown("#### Line breakdown")
        df_break = pd.DataFrame(result["breakdown"])
        st.dataframe(df_break, use_container_width=True, hide_index=True)
        st.download_button(
            "Download breakdown (CSV)",
            data=df_break.to_csv(index=False).encode("utf-8"),
            file_name="breakdown_pro.csv",
            mime="text/csv",
        )

        st.markdown("#### Audit trail")
        if result["exclusive_name"]:
            st.info(f"Exclusive rule in effect: **{result['exclusive_name']}**")
        if result["audit"]:
            for a in result["audit"]:
                st.write("• " + a)
        else:
            st.write("No rules applied.")

# ----------------------------- Help Tab ---------------------------------
with tab_help:
    st.subheader("How this maps to D365 FO Retail")
    st.markdown(
        """
**Price groups**: Rules link to price groups. A rule is eligible when any of its price groups
intersects with the store or customer price groups (similar to D365 linkage).

**Store hierarchy**: Target specific stores or store groups. Store groups approximate hierarchy nodes.

**Validation periods**: Date range + optional day-of-week and time windows (local).

**Customer context**: Employee flag, affiliations, and customer price groups can further restrict rules.

This is still a simulator for learning & what-if testing—validate final behavior in D365 test.
"""
    )
