from confledger.utility.answer_parsing import parse_answer, parse_confidence, parse_json_list


def test_answers():
    assert parse_answer("Answer: Yes\nExplanation: x") == "yes"
    assert parse_answer("**Answer:** no, because") == "no"
    assert parse_answer("Yes, since the capital is Paris.") == "yes"
    assert parse_answer("Answer: B", "ab") == "yes" and parse_answer("Answer: A", "ab") == "no"
    assert parse_answer("It depends.") is None


def test_confidence_and_json():
    assert parse_confidence("Answer: Yes\nConfidence: 85%") == 85
    assert parse_confidence("Confidence: 300") is None
    assert parse_json_list('```json\n[0, 1, 0]\n```') == [0, 1, 0]
    assert parse_json_list("no list here") is None
