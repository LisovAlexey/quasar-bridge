package bridge

import (
	"context"
	"testing"

	"quasarbridge/internal/llm"
)

// TestClaimForcesLLMForOneTurn checks that a daemon control claim overrides text
// routing for exactly one turn, and that the turn after it routes normally again
// (so a following "Алиса …" is never captured). Continuity across turns comes
// from short-term memory, not a sticky route.
func TestClaimForcesLLMForOneTurn(t *testing.T) {
	e := &Engine{}
	if err := e.SetRouting(DefaultRoutingConfig()); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	// Baseline: an Alice-domain query with no claim goes to Yandex by rule.
	if d := e.Decide(ctx, "какая погода"); d.Route != "yandex" {
		t.Fatalf("baseline route = %+v, want yandex", d)
	}

	// A claim forces the next turn to the LLM regardless of the recognized text.
	e.Claim(false)
	if d := e.Decide(ctx, "какая погода"); d.Route != "llm" || d.Source != "control_claim" {
		t.Fatalf("claimed turn = %+v, want llm/control_claim", d)
	}

	// The turn after the claim routes normally again: an Alice-domain query here
	// (as after a fresh "Алиса" wake) must go to Yandex, not be captured.
	if d := e.Decide(ctx, "какая погода"); d.Route != "yandex" {
		t.Fatalf("post-claim route = %+v, want yandex", d)
	}
}

// TestClaimExpires confirms a stale claim does not force the LLM.
func TestClaimExpires(t *testing.T) {
	e := &Engine{}
	if err := e.SetRouting(DefaultRoutingConfig()); err != nil {
		t.Fatal(err)
	}
	e.Claim(false)
	e.claimUntil.Store(1) // far in the past
	if d := e.Decide(context.Background(), "какая погода"); d.Source == "control_claim" {
		t.Fatalf("expired claim should not force llm, got %+v", d)
	}
}

// TestAliceBetweenClaudeKeepsMemory checks that an Alice (yandex) turn between two
// Claude turns does not wipe the Claude chat: the second "Клод" turn still sees the
// first question and answer in its context.
func TestAliceBetweenClaudeKeepsMemory(t *testing.T) {
	calls := 0
	e := &Engine{LLM: completionFunc(func(_ context.Context, m []llm.Message, structured bool) (string, error) {
		if structured {
			return `{"route":"yandex"}`, nil
		}
		calls++
		if calls == 2 {
			if len(m) != 4 || m[1].Content != "спроси нейросеть: о звёздах" || m[2].Content != "ответ клода" {
				t.Errorf("second Claude turn lost memory: %+v", m)
			}
		}
		return "ответ клода", nil
	})}
	if r, err := e.Resolve(t.Context(), "спроси нейросеть: о звёздах"); err != nil || r.Route != "llm" {
		t.Fatal(r, err)
	}
	if r, err := e.Resolve(t.Context(), "какая погода сегодня"); err != nil || r.Route != "yandex" {
		t.Fatal(r, err)
	}
	if r, err := e.Resolve(t.Context(), "спроси нейросеть: второй вопрос"); err != nil || r.Route != "llm" || calls != 2 {
		t.Fatalf("second Claude turn: %+v err=%v calls=%d", r, err, calls)
	}
}
