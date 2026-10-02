from laya import Router

router = Router()  # downloads a checkpoint on first use; Router(preload=True) loads all three up front

state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
questions = {
    "department": {"type": "choice", "instructions": "Which department should handle `message`?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is `message`?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does `message` threaten to cancel or leave?"},
    "sentiment": {"type": "score", "instructions": "How unhappy is the user in `message`?",
                  "criteria": ["neutral", "annoyed", "angry", "furious"]},
    "refund_requested": {"type": "noul", "instructions": "Does `message` ask for money back?",
                         "criteria": {"true": "asks for a refund or a charge reversed",
                                      "false": "no refund requested"}},
    "deadline_mentioned": {"type": "noul",
                           "instructions": "Does `message` name a deadline or time pressure?"},
    "security_related": {"type": "noul",
                         "instructions": "Does `message` report fraud, account compromise or a data breach?"},
}


def main() -> None:
    result = router.predict(state, questions)

    print(result["answers"]["department"]["choice"])            # billing
    print(result["answers"]["department"]["answer_confidence"])  # 0.9875 -- max(p), the calibrated field
    print(result["answers"]["urgency"]["score"])                # 1.79 -- expected level on the 0..2 scale
    print(result["answers"]["churn_risk"]["noul"])              # 0.84 -- P(the answer is yes)
    print(result["answers"]["sentiment"]["score"])              # 1.64 -- expected level on the 0..3 scale
    print(result["answers"]["refund_requested"]["noul"])        # 0.91
    print(result["answers"]["deadline_mentioned"]["noul"])      # 0.24 -- "today" not read as a deadline
    print(result["answers"]["security_related"]["noul"])        # 0.37 -- no fraud or breach reported
    print(result["routing"]["model"])                           # english


if __name__ == "__main__":
    main()
