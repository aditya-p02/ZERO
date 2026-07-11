from main import _is_exit


def test_exit_commands_are_detected():
    assert _is_exit("exit")
    assert _is_exit("zero shutdown")
    assert _is_exit("bye")


def test_exit_words_do_not_trigger_inside_unrelated_questions():
    assert not _is_exit("how do i exit vim")
    assert not _is_exit("can you shutdown a process")
    assert not _is_exit("what are exit codes")
    assert not _is_exit("explain recursion")
